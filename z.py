import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# --- Ensure local modules can be found ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "stroke_classification"))
from model.bst_3d import BST_CG_AP
from preparing_data.shuttleset_dataset import get_stroke_types

# --- Configuration & Constants ---
NORM_W, NORM_H = 1280.0, 720.0
COURT_W, COURT_L, COURT_H = 6.1, 13.4, 5.0 
COCO_BONES = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
              (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

ALL_TYPES = get_stroke_types()

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
                digits = re.findall(r'\d+', count_str)
                if digits: data[match_name.strip()] = int(digits[-1])
    return data

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    root_dir = Path("test_shuttlenet")
    gt_dir = Path("shuttleset/set")
    counts_path = Path("video_frame_counts.txt")
    weight_file = Path("weights/checkpoints/latest_global_checkpoint.pt")

    print(f"\n{'='*70}")
    print(f" GLOBAL EVALUATION PIPELINE")
    print(f"{'='*70}")

    if not root_dir.exists() or not gt_dir.exists():
        print("❌ Error: Could not find 'test_shuttlenet' or 'shuttleset/set' folders.")
        return

    if not weight_file.exists():
        print(f"❌ Error: Checkpoint not found at {weight_file}.")
        return

    match_properties = load_video_frame_counts(counts_path)

    # 1. Load Model Once
    print("⏳ Loading model and weights into memory...")
    model = BST_CG_AP(in_dim=72, seq_len=100, n_class=35).to(device)
    checkpoint = torch.load(weight_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Tracking Variables
    total_correct = 0
    total_shots = 0
    match_results = {}

    match_folders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
    
    for match_folder in match_folders:
        match_name = match_folder.name
        csv_dir = gt_dir / match_name
        
        n_total = 1000000

        # Skip invalid or incomplete folders silently
        if not csv_dir.exists() or n_total is None:
            continue

        traj_path = match_folder / "traj_out" / f"{match_name}_traj_3d_trimmed.npy"
        pose_path = match_folder / "pose_out" / "poses.pkl"
        if not traj_path.exists() or not pose_path.exists():
            continue

        print(f"\n🎯 Evaluating: {match_name[:40]}...")
        
        # Load Arrays
        traj_3d = np.load(traj_path)
        with open(pose_path, "rb") as f: poses = pickle.load(f)

        # Parse Ground Truth
        dfs = []
        for csv_file in csv_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            df['set_file'] = csv_file.name
            dfs.append(df)
        
        if not dfs: continue
            
        gt_df = pd.concat(dfs).dropna(subset=["frame_num"])
        gt_df["frame_num"] = gt_df["frame_num"].astype(int)
        gt_df = gt_df.sort_values(by=["set_file", "rally", "frame_num"])

        required_frames = []
        for _, rally_group in gt_df.groupby(["set_file", "rally"]):
            f_start, f_end = int(rally_group["frame_num"].min()), int(rally_group["frame_num"].max())
            for f in range(max(0, f_start - 60), min(n_total, f_end + 61)):
                required_frames.append(f)
        trim_lookup = {orig_f: idx for idx, orig_f in enumerate(required_frames)}

        # Extract Features
        X_jnb, X_shuttle, X_pos, Y_labels = [], [], [], []
        for _, rally_group in gt_df.groupby("rally"):
            hits = rally_group.sort_values("frame_num").to_dict('records')
            for i in range(len(hits)):
                hit_n = int(hits[i]["frame_num"])
                hit_prev = int(hits[i-1]["frame_num"]) if i > 0 else 0
                hit_next = int(hits[i+1]["frame_num"]) if i < len(hits) - 1 else n_total - 1
                
                py, oy = hits[i].get('player_location_y', 0), hits[i].get('opponent_location_y', 0)
                side = 'Top' if py < oy else 'Bottom'
                stroke_col = 'type' if 'type' in hits[i] else 'stroke_type'
                full_label = f"{side}_{hits[i].get(stroke_col, '')}"
                
                if full_label in ALL_TYPES:
                    label_idx = ALL_TYPES.index(full_label)
                    start_f = max(0, max(hit_prev, hit_n - 50))
                    end_f = min(n_total - 1, min(hit_next, hit_n + 49))
                    
                    jnb, pos, shut = extract_features(poses, start_f, end_f, hit_n, traj_3d, trim_lookup)
                    X_jnb.append(jnb); X_shuttle.append(shut); X_pos.append(pos)
                    Y_labels.append(label_idx)

        if not Y_labels:
            print("  ↳ No valid labels found. Skipping.")
            continue

        # Evaluate this match
        dataset = Badminton3DDataset(X_jnb, X_shuttle, X_pos, Y_labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for (j_batch, p_batch, s_batch), vlen, l_batch in loader:
                logits = model(j_batch.to(device), s_batch.to(device), p_batch.to(device), vlen.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(l_batch.numpy())

        # Match Results
        match_shots = len(all_labels)
        match_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        match_acc = match_correct / match_shots if match_shots > 0 else 0

        total_shots += match_shots
        total_correct += match_correct
        match_results[match_name] = match_acc

        print(f"  ↳ Shots: {match_shots} | Accuracy: {match_acc * 100:.2f}%")

    # --- FINAL SUMMARY ---
    print(f"\n{'='*70}")
    print(f" FINAL GLOBAL RESULTS ")
    print(f"{'='*70}")
    
    if total_shots == 0:
        print("❌ No videos were successfully processed.")
        return

    print(f" 📂 Total Videos Evaluated : {len(match_results)}")
    print(f" 🏸 Total Shots Evaluated  : {total_shots}")
    
    # Macro Average (Average of the per-video percentages)
    macro_acc = sum(match_results.values()) / len(match_results)
    
    # Micro Average (Total Correct / Total Shots)
    micro_acc = total_correct / total_shots

    print(f" 📊 Average Video Accuracy : {macro_acc * 100:.2f}%")
    print(f" 🏆 True Global Accuracy   : {micro_acc * 100:.2f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()