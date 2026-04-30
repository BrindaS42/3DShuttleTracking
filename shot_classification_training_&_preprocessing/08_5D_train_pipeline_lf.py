import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

# --- Path Configuration ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "stroke_classification"))
from model.bst_5d_lf import BST_CG_AP # Importing the code above
from preparing_data.shuttleset_dataset import get_stroke_types

class Badminton5DDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f: data = pickle.load(f)
        self.jnb = torch.tensor(np.array(data['jnb']), dtype=torch.float32)
        self.pos = torch.tensor(np.array(data['pos']), dtype=torch.float32)
        # Fusing 2D (u,v) and 3D (x,y,z) coordinates
        self.shut = torch.tensor(np.concatenate([np.array(data['s2d']), np.array(data['s3d'])], axis=-1), dtype=torch.float32)
        self.labels = torch.tensor(np.array(data['labels']), dtype=torch.long)
        self.v_len = torch.full((len(self.labels),), 100, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return (self.jnb[i], self.shut[i], self.pos[i]), self.v_len[i], self.labels[i]

@torch.no_grad()
def evaluate_comprehensive(model, loader, device, n_classes=35):
    model.eval()
    all_p, all_l, all_logit = [], [], []
    for (j, s, p), v, l in loader:
        logits = model(j.to(device), s.to(device), p.to(device), v.to(device))
        all_logit.append(logits.cpu()); all_p.extend(torch.argmax(logits, 1).cpu().numpy()); all_l.extend(l.numpy())
    all_logit = torch.cat(all_logit)
    l_t = torch.tensor(all_l)
    acc = accuracy_score(all_l, all_p)
    _, top2 = torch.topk(all_logit, 2, 1)
    acc2 = torch.any(top2 == l_t.unsqueeze(1), 1).float().mean().item()
    f1 = f1_score(all_l, all_p, average=None, labels=np.arange(n_classes), zero_division=0)
    return {"acc": acc, "acc2": acc2, "macro_f1": np.mean(f1), "min_f1": np.min(f1[np.unique(all_l)]) if len(np.unique(all_l)) > 0 else 0.0}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_dir, ckpt_dir = Path("weights/dataset_cache"), Path("weights/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_wt, resume_ckpt = ckpt_dir / "best_bst_5d_global_lf.pt", ckpt_dir / "latest_5d_checkpoint_lf.pt"

    if not (cache_dir / "train_data.pkl").exists():
        print("❌ Run 08_prepare_global_dataset.py first."); return

    train_ldr = DataLoader(Badminton5DDataset(cache_dir/"train_data.pkl"), batch_size=64, shuffle=True)
    val_ldr = DataLoader(Badminton5DDataset(cache_dir/"val_data.pkl"), batch_size=64)
    test_ldr = DataLoader(Badminton5DDataset(cache_dir/"test_data.pkl"), batch_size=64)

    model = BST_CG_AP(in_dim=72, seq_len=100, n_class=35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    start_ep, best_v_acc, early_stop_patience, early_stop_ctr = 1, 0.0, 25, 0
    if resume_ckpt.exists():
        ck = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ck['model_state_dict']); optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_ep, best_v_acc = ck['epoch'] + 1, ck.get('best_val_acc', 0.0)
        print(f"🔄 Resumed from Ep {start_ep}")

    total_ep = 150
    scheduler = get_cosine_schedule_with_warmup(optimizer, 400, total_ep * len(train_ldr))

    for ep in range(start_ep, total_ep + 1):
        model.train()
        loss_total = 0.0
        pbar = tqdm(train_ldr, desc=f"5D Ep {ep:03d}", leave=False)
        for (j, s, p), v, l in pbar:
            optimizer.zero_grad()
            out = model(j.to(device), s.to(device), p.to(device), v.to(device))
            loss = criterion(out, l.to(device))
            loss.backward(); optimizer.step(); scheduler.step()
            loss_total += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        m = evaluate_comprehensive(model, val_ldr, device)
        if m['acc'] > best_v_acc:
            best_v_acc, early_stop_ctr = m['acc'], 0
            torch.save({'model_state_dict': model.state_dict(), 'metrics': m}, best_wt)
            print(f"⭐ [New Best] Acc: {m['acc']:.4f} | Acc@2: {m['acc2']:.4f} | mF1: {m['macro_f1']:.4f}")
        else: early_stop_ctr += 1
            
        print(f"📊 [Val] Ep {ep:03d} | Loss: {loss_total/len(train_ldr):.4f} | Acc: {m['acc']:.4f} | Acc@2: {m['acc2']:.4f} | mF1: {m['macro_f1']:.4f} | minF1: {m['min_f1']:.4f} | Patience: {early_stop_ctr}/{early_stop_patience}")
        if ep % 10 == 0: torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_acc': best_v_acc}, resume_ckpt)
        if early_stop_ctr >= early_stop_patience: print(f"\n🛑 Early Stopping at epoch {ep}."); break

    print(f"\n{'='*70}\n🏁 FINAL 5D (FUSED) TEST RESULTS")
    model.load_state_dict(torch.load(best_wt)['model_state_dict'])
    tm = evaluate_comprehensive(model, test_ldr, device)
    print(f"  Accuracy (Top-1) : {tm['acc']*100:.2f}% | Top-2: {tm['acc2']*100:.2f}%")
    print(f"  Macro-F1: {tm['macro_f1']:.4f} | Min-F1: {tm['min_f1']:.4f}\n{'='*70}")

if __name__ == "__main__": main()