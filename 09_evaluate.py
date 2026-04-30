import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# --- Path Configuration ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "stroke_classification"))

# Import architectures
try:
    from model.bst_2d import BST_CG_AP as BST_2D
    from model.bst_5d import BST_CG_AP as BST_5D_EF
    from model.bst_5d_lf import BST_CG_AP as BST_5D_LF
    from preparing_data.shuttleset_dataset import get_stroke_types
except ImportError as e:
    print(f"❌ Architecture import failed: {e}")

# Global Class List for side-matching logic
ALL_TYPES = get_stroke_types()

class BadmintonEvalDataset(Dataset):
    def __init__(self, data_path, mode='2d'):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.jnb = torch.tensor(np.array(data['jnb']), dtype=torch.float32)
        self.pos = torch.tensor(np.array(data['pos']), dtype=torch.float32)
        
        if mode == '2d':
            self.shut = torch.tensor(np.array(data['s2d']), dtype=torch.float32)
        elif mode == '5d':
            s2d, s3d = np.array(data['s2d']), np.array(data['s3d'])
            self.shut = torch.tensor(np.concatenate([s2d, s3d], axis=-1), dtype=torch.float32)
            
        self.labels = torch.tensor(np.array(data['labels']), dtype=torch.long)
        self.v_len = torch.full((len(self.labels),), 100, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return (self.jnb[i], self.shut[i], self.pos[i]), self.v_len[i], self.labels[i]

@torch.no_grad()
def run_evaluation_constrained(model, loader, device, n_classes=35):
    """
    Calculates accuracy with Side-Constrained Post-Processing: 
    If the top prediction side differs from Ground Truth side, 
    choose the next best prediction that matches the side.
    """
    model.eval()
    all_preds_post, all_labels = [], []
    
    for (j, s, p), v, labels in tqdm(loader, desc="Evaluating (Side-Constrained)", leave=False):
        logits = model(j.to(device), s.to(device), p.to(device), v.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        labels_batch = labels.numpy()
        
        for b_idx in range(len(labels_batch)):
            gt_idx = labels_batch[b_idx]
            gt_label_str = ALL_TYPES[gt_idx]
            # Determine Ground Truth side
            gt_side = gt_label_str.split("_", 1)[0] if "_" in gt_label_str else "Unknown"
            
            # Sort all class probabilities for this sample
            cand_indices = np.argsort(probs[b_idx])[::-1]
            
            final_pred_idx = cand_indices[0] # Default to top-1
            
            # SIDE-CONSTRAINED LOGIC: Find next best accurate prediction with matched hitter type
            if gt_side in ["Top", "Bottom"]:
                for cand_idx in cand_indices:
                    cand_str = ALL_TYPES[cand_idx]
                    cand_side = cand_str.split("_", 1)[0] if "_" in cand_str else "Unknown"
                    
                    if cand_side == gt_side:
                        final_pred_idx = cand_idx
                        break
            
            all_preds_post.append(final_pred_idx)
            all_labels.append(gt_idx)
    
    # Calculate metrics based on side-corrected predictions
    acc = accuracy_score(all_labels, all_preds_post)
    f1 = f1_score(all_labels, all_preds_post, average=None, labels=np.arange(n_classes), zero_division=0)
    
    return {
        "acc": acc * 100,
        "mf1": np.mean(f1),
        "minf1": np.min(f1[np.unique(all_labels)]) if len(np.unique(all_labels)) > 0 else 0.0
    }

def main():
    # HF free tier/deployment requires CPU
    device = torch.device('cpu') 
    cache_dir = Path("weights/dataset_cache")
    ckpt_dir = Path("weights/checkpoints")

    configs = [
        ("best_bst_2d_global.pt", BST_2D, '2d'),
        ("best_bst_5d_global.pt", BST_5D_EF, '5d'),
        ("best_bst_5d_global_lf.pt", BST_5D_LF, '5d')
    ]

    results = []

    for ckpt_name, arch_class, mode in configs:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists(): continue

        print(f"\n🔍 Evaluating with Side-Constraint: {ckpt_name}")
        model = arch_class(in_dim=72, seq_len=100, n_class=35).to(device)
        # Using weights_only=False to support NumPy scalars in existing checkpoints
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state))

        for split in ['test']: # Focus on final performance
            data_file = cache_dir / f"{split}_data.pkl"
            if not data_file.exists(): continue
            
            loader = DataLoader(BadmintonEvalDataset(data_file, mode), batch_size=64)
            m = run_evaluation_constrained(model, loader, device)
            results.append({
                "Model": ckpt_name,
                "Acc (Post-Proc)": f"{m['acc']:.2f}%",
                "mF1": f"{m['mf1']:.4f}",
                "minF1": f"{m['minf1']:.4f}"
            })

    # --- Print Summary Table ---
    print("\n" + "="*70)
    print(f"{'Checkpoint':<30} | {'Acc (Post)':<12} | {'mF1':<7} | {'minF1'}")
    print("-" * 70)
    for r in results:
        print(f"{r['Model']:<30} | {r['Acc (Post-Proc)']:<12} | {r['mF1']:<7} | {r['minF1']}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()