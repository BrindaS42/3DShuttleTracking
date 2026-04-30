import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from copy import deepcopy
from tqdm import tqdm # Required for progress bars

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

# --- FIX: Ensure local modules can be found ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "stroke_classification"))
from model.bst_3d import BST_CG_AP
from preparing_data.shuttleset_dataset import get_stroke_types

# --- Configuration & Hardcoded Constants ---
NORM_W, NORM_H = 640.0, 360.0 
COURT_W, COURT_L, COURT_H = 6.1, 13.4, 5.0 
COCO_BONES = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
              (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

ALL_TYPES = get_stroke_types()


def get_stroke_name_mapping():
    return {
        "擋小球": "Return Net", "殺球": "Smash", "點扣": "Wrist Smash", "挑球": "Lob",
        "防守回挑": "Defensive Return Lob", "長球": "Clear", "平球": "Drive",
        "後場抽平球": "Back-court Drive", "切球": "Drop", "過渡切球": "Passive Drop",
        "推球": "Push", "撲球": "Rush (Kill)", "防守回抽": "Defensive Return Drive",
        "勾球": "Cross-court Net Shot", "發短球": "Short Service", "發長球": "Long Service",
        "放小球": "Net Shot", "未知球種": "Unknown", "None": "None/Transition"
    }
# --- Helper Functions ---

def load_video_frame_counts(filepath):
    data = {}
    if not filepath.exists(): return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("video_name"): continue
            parts = line.split()
            if len(parts) >= 2:
                count_str = parts[-1]
                match_name = " ".join(parts[:-1])
                if match_name.endswith(".mp4"): match_name = match_name[:-4]
                data[match_name.strip()] = int(count_str)
    return data

def extract_features(poses, start_f, end_f, hit_n, traj_3d, trim_lookup):
    jnb_seq = np.zeros((100, 2, 72), dtype=np.float32)
    pos_seq = np.zeros((100, 2, 2), dtype=np.float32)
    shuttle_seq = np.zeros((100, 3), dtype=np.float32)
    seq_start = 50 + (start_f - hit_n)
    
    for i, orig_f in enumerate(range(start_f, end_f + 1)):
        t_idx = seq_start + i
        if t_idx < 0 or t_idx >= 100: continue
        trim_idx = trim_lookup.get(orig_f, -1)
        if trim_idx == -1 or trim_idx >= len(poses): continue
        raw_shuttle = traj_3d[trim_idx]
        shuttle_seq[t_idx] = np.nan_to_num(raw_shuttle) / [COURT_W, COURT_L, COURT_H]
        pf = poses[trim_idx]
        for p_idx, p_attr in enumerate(['near', 'far']):
            player = getattr(pf, p_attr, None)
            if player and player.keypoints:
                kpts = np.array(player.keypoints)[:, :2] / [NORM_W, NORM_H]
                kpts = np.clip(kpts, 0, 1)
                bones = np.zeros((19, 2))
                for b_idx, (j1, j2) in enumerate(COCO_BONES):
                    bones[b_idx] = kpts[j1] - kpts[j2]
                jnb_seq[t_idx, p_idx, :34] = kpts.flatten()
                jnb_seq[t_idx, p_idx, 34:72] = bones[:19].flatten()
                if player.floor_pos_3d is not None:
                    f_pos = np.nan_to_num(player.floor_pos_3d[:2])
                    pos_seq[t_idx, p_idx] = f_pos / [COURT_W, COURT_L]
    return jnb_seq, pos_seq, shuttle_seq

class Badminton3DDataset(Dataset):
    def __init__(self, jnb, shuttle, pos, labels):
        self.jnb = torch.tensor(np.array(jnb), dtype=torch.float32)
        self.shuttle = torch.tensor(np.array(shuttle), dtype=torch.float32)
        self.pos = torch.tensor(np.array(pos), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.v_len = torch.full((len(self.labels),), 100, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.jnb[idx], self.pos[idx], self.shuttle[idx]), self.v_len[idx], self.labels[idx]

@torch.no_grad()
def evaluate_comprehensive(model, loader, device, n_classes=35):
    """Computes Acc, Acc@2, MacroF1, and MinF1."""
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    
    for (jnb, pos, shut), vlen, labels in loader:
        logits = model(jnb.to(device), shut.to(device), pos.to(device), vlen.to(device))
        all_logits.append(logits.cpu())
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    
    all_logits = torch.cat(all_logits)
    labels_tensor = torch.tensor(all_labels)
    
    # 1. Top-1 Accuracy
    acc = accuracy_score(all_labels, all_preds)
    
    # 2. Top-2 Accuracy
    _, top2_preds = torch.topk(all_logits, k=2, dim=1)
    acc2 = torch.any(top2_preds == labels_tensor.unsqueeze(1), dim=1).float().mean().item()
    
    # 3. F1 Metrics
    # Using 'zero_division=0' to handle classes with no instances in test set
    f1_each = f1_score(all_labels, all_preds, average=None, labels=np.arange(n_classes), zero_division=0)
    macro_f1 = np.mean(f1_each)
    
    # Min F1 (ignoring absolute zeros which represent classes not present in test slice)
    present_mask = np.unique(all_labels)
    min_f1 = np.min(f1_each[present_mask]) if len(present_mask) > 0 else 0.0
    
    return {"acc": acc, "acc2": acc2, "macro_f1": macro_f1, "min_f1": min_f1}

def print_global_distribution(Y_train, Y_test):
    """Prints a detailed table of stroke counts for Train and Test sets."""
    train_counts = Counter(Y_train)
    test_counts = Counter(Y_test)
    c2e = get_stroke_name_mapping()
    
    print("\n" + "="*85)
    print(f"{'ID':<4} {'Stroke Type (Side)':<45} {'Train':<10} {'Test':<10}")
    print("-" * 85)
    for i in range(35):
        full_name = ALL_TYPES[i]
        side, chi = full_name.split("_", 1) if "_" in full_name else ("", full_name)
        eng = f"({side}) {c2e.get(chi, chi)}"
        print(f"{i:<4} {eng:<45} {train_counts.get(i, 0):<10} {test_counts.get(i, 0):<10}")
    print("="*85 + "\n")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir, counts_path = Path("test_shuttlenet"), Path("video_frame_counts.txt")
    gt_dir, ckpt_dir = Path("shuttleset/set"), Path("weights/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_weight_file = ckpt_dir / "global_best_bst_3d.pt"
    cache_file = ckpt_dir / "global_data_cache.pkl"
    if cache_file.exists():
        print(f"📂 Loading extracted features from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            data_dict = pickle.load(f)
            G_jnb, G_shuttle, G_pos, G_labels = data_dict['jnb'], data_dict['shuttle'], data_dict['pos'], data_dict['labels']
    else:
        print("⏳ Cache not found. Extracting Global Dataset...")
        G_jnb, G_shuttle, G_pos, G_labels = [], [], [], []
        match_properties = load_video_frame_counts(counts_path)
        
        # --- TRACKING DICTIONARY ---
        video_stats = {} 

        match_folders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
        print(f"🔍 Found {len(match_folders)} folders in {root_dir}")

        for match_folder in tqdm(match_folders, desc="Extracting Global Dataset"):
            match_name = match_folder.name
            # n_total = next((v for k, v in match_properties.items() if k.replace(" ","") == match_name.replace(" ","")), None)
            n_total = 1000000
            traj_path = match_folder / "traj_out" / f"{match_name}_traj_3d_trimmed.npy"
            pose_path = match_folder / "pose_out" / "poses.pkl"
            csv_dir = gt_dir / match_name
            
            # Check which specific file is missing for debugging
            missing = []
            if not traj_path.exists(): missing.append("3D Traj")
            if not pose_path.exists(): missing.append("Poses")
            if not csv_dir.exists(): missing.append("CSV Dir")
            if not n_total: missing.append("Frame Count Entry")
            
            if missing:
                print(f"⚠️ Skipping {match_name}: Missing {', '.join(missing)}")
                continue
            
            traj_3d = np.load(traj_path)
            with open(pose_path, "rb") as f: poses = pickle.load(f)
            dfs = [pd.read_csv(f).assign(set_file=f.name) for f in csv_dir.glob("*.csv")]
            if not dfs: continue
            gt_df = pd.concat(dfs).dropna(subset=["frame_num"])
            gt_df["frame_num"] = gt_df["frame_num"].astype(int)
            
            # Count actual strokes in CSV
            actual_hits_in_csv = len(gt_df)
            processed_hits = 0
            
            required_frames = []
            for _, rally_group in gt_df.groupby(["set_file", "rally"]):
                f_start, f_end = int(rally_group["frame_num"].min()), int(rally_group["frame_num"].max())
                required_frames.extend(range(max(0, f_start-60), min(n_total, f_end+61)))
            trim_lookup = {orig_f: idx for idx, orig_f in enumerate(required_frames)}

            for _, rally_group in gt_df.groupby("rally"):
                hits = rally_group.sort_values("frame_num").to_dict('records')
                for i in range(len(hits)):
                    hit_n = int(hits[i]["frame_num"])
                    py, oy = hits[i].get('player_location_y', 0), hits[i].get('opponent_location_y', 0)
                    full_label = f"{'Top' if py < oy else 'Bottom'}_{hits[i].get('type', hits[i].get('stroke_type', ''))}"
                    
                    if full_label in ALL_TYPES:
                        start_f = max(0, max(int(hits[i-1]["frame_num"]) if i > 0 else 0, hit_n - 50))
                        end_f = min(n_total - 1, min(int(hits[i+1]["frame_num"]) if i < len(hits)-1 else n_total-1, hit_n + 49))
                        jnb, pos, shut = extract_features(poses, start_f, end_f, hit_n, traj_3d, trim_lookup)
                        G_jnb.append(jnb); G_shuttle.append(shut); G_pos.append(pos); G_labels.append(ALL_TYPES.index(full_label))
                        processed_hits += 1

            video_stats[match_name] = {"csv": actual_hits_in_csv, "processed": processed_hits}
            
        print("\n" + "="*50)
        print(f"{'Video Name':<30} | {'CSV':<5} | {'Final'}")
        print("-" * 50)
        total_csv, total_proc = 0, 0
        for name, stat in video_stats.items():
            print(f"{name[:30]:<30} | {stat['csv']:<5} | {stat['processed']}")
            total_csv += stat['csv']
            total_proc += stat['processed']
        print("-" * 50)
        print(f"{'TOTAL':<30} | {total_csv:<5} | {total_proc}")
        print("="*50 + "\n")

        if total_csv != total_proc:
            print(f"💡 Note: {total_csv - total_proc} strokes were dropped (likely 'Unknown' types or out-of-bounds frames).")
        # After loop:
        with open(cache_file, "wb") as f:
            pickle.dump({'jnb': G_jnb, 'shuttle': G_shuttle, 'pos': G_pos, 'labels': G_labels}, f)
        print(f"✅ Features cached to {cache_file}")
    

    # --- SPLITTING ---
    X_j_tr, X_j_temp, X_s_tr, X_s_temp, X_p_tr, X_p_temp, Y_tr, Y_temp = train_test_split(
        G_jnb, G_shuttle, G_pos, G_labels, test_size=0.20, stratify=G_labels, random_state=42
    )
    X_j_v, X_j_te, X_s_v, X_s_te, X_p_v, X_p_te, Y_v, Y_te = train_test_split(
        X_j_temp, X_s_temp, X_p_temp, Y_temp, test_size=0.50, stratify=Y_temp, random_state=42
    )

    # --- TRAINING ---
    batch_size = 64
    train_loader = DataLoader(Badminton3DDataset(X_j_tr, X_s_tr, X_p_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Badminton3DDataset(X_j_v, X_s_v, X_p_v, Y_v), batch_size=batch_size)
    test_loader  = DataLoader(Badminton3DDataset(X_j_te, X_s_te, X_p_te, Y_te), batch_size=batch_size)

    print_global_distribution(Y_tr, Y_te)

    model = BST_CG_AP(in_dim=72, seq_len=100, n_class=35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # label_counts = Counter(Y_tr)
    # weights = torch.tensor([1.0 / (label_counts[i] + 1e-6) for i in range(35)], dtype=torch.float32).to(device)
    # weights = weights / weights.sum() * 35 

    total_epochs = 150
    start_epoch = 1
    best_val_acc = 0.0
    best_weights = None

    early_stop_patience = 25  # Stop if no improvement in 25 epochs
    early_stop_counter = 0

    # --- RESUME FROM CHECKPOINT IF EXISTS ---
    periodic_path = ckpt_dir / "latest_global_checkpoint.pt"
    if periodic_path.exists():
        print(f"🔄 Resuming training from existing checkpoint: {periodic_path}")
        checkpoint = torch.load(periodic_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"📈 Resuming from Epoch {start_epoch} | Previous Best Acc: {best_val_acc:.4f}")
    else:
        print("🆕 No checkpoint found. Starting fresh with default initialization.")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_epochs = 150
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=400, 
        num_training_steps=total_epochs * len(train_loader),
        num_cycles=0.5 # Adds a "second chance" for the model to learn
    )


    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_loss = 0
        
        # Batch Progress Bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{total_epochs}", leave=False)
        for (j_batch, p_batch, s_batch), vlen, l_batch in pbar:
            optimizer.zero_grad()
            output = model(j_batch.to(device), s_batch.to(device), p_batch.to(device), vlen.to(device))
            loss = criterion(output, l_batch.to(device))
            loss.backward(); optimizer.step(); scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        
        metrics = evaluate_comprehensive(model, val_loader, device)
        is_best = metrics['acc'] >= best_val_acc
        if is_best:
            best_val_acc = metrics['acc']
            best_weights = deepcopy(model.state_dict())

        if metrics['acc'] > best_val_acc:
            early_stop_counter = 0 
        else:
            early_stop_counter += 1
            
        print(f"📊 [Val] Ep {epoch:03d} | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Acc: {metrics['acc']:.4f} | Acc@2: {metrics['acc2']:.4f} | "
                  f"mF1: {metrics['macro_f1']:.4f} | minF1: {metrics['min_f1']:.4f} {'⭐' if is_best else ''}")

            # --- PERIODIC CHECKPOINT (Every 10 Epochs) ---
        if epoch % 10 == 0 or early_stop_counter == 0:
            periodic_path = ckpt_dir / "latest_global_checkpoint.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, periodic_path)
            print(f"💾 Periodic checkpoint saved to {periodic_path}")

        # if early_stop_counter >= early_stop_patience:
        #     print(f"\n🛑 EARLY STOPPING triggered at epoch {epoch}. No improvement for {early_stop_patience} epochs.")
        #     break

    # --- FINAL TEST ---
    if best_weights:
        model.load_state_dict(best_weights)
        torch.save({'model_state_dict': best_weights}, global_weight_file)

    test_metrics = evaluate_comprehensive(model, test_loader, device)
    print(f"\n{'='*70}\n🏆 FINAL TEST RESULTS (GLOBAL MIXED DATASET)")
    print(f"  Accuracy (Top-1) : {test_metrics['acc']*100:.2f}%")
    print(f"  Accuracy (Top-2) : {test_metrics['acc2']*100:.2f}%")
    print(f"  Macro-F1 Score   : {test_metrics['macro_f1']:.4f}")
    print(f"  Min-F1 Score     : {test_metrics['min_f1']:.4f}\n{'='*70}")

if __name__ == "__main__": main()