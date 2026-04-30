import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

# --- FIX: Ensure local modules can be found ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "stroke_classification"))
from model.bst_2d import BST_CG_AP
from preparing_data.shuttleset_dataset import get_stroke_types

# --- Constants ---
ALL_TYPES = get_stroke_types()

class Badminton2DDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.jnb = torch.tensor(np.array(data['jnb']), dtype=torch.float32)
        self.pos = torch.tensor(np.array(data['pos']), dtype=torch.float32)
        # Using 2D pixel-based shuttle coordinates
        self.shut = torch.tensor(np.array(data['s2d']), dtype=torch.float32)
        self.labels = torch.tensor(np.array(data['labels']), dtype=torch.long)
        self.v_len = torch.full((len(self.labels),), 100, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.jnb[i], self.pos[i], self.shut[i]), self.v_len[i], self.labels[i]

@torch.no_grad()
def evaluate_comprehensive(model, loader, device, n_classes=35):
    """Calculates all requested metrics: Acc, Acc@2, MacroF1, and MinF1."""
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
    f1_each = f1_score(all_labels, all_preds, average=None, labels=np.arange(n_classes), zero_division=0)
    macro_f1 = np.mean(f1_each)
    
    # Min F1 (ignoring classes not present in this specific split)
    present_mask = np.unique(all_labels)
    min_f1 = np.min(f1_each[present_mask]) if len(present_mask) > 0 else 0.0
    
    return {"acc": acc, "acc2": acc2, "macro_f1": macro_f1, "min_f1": min_f1}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_dir = Path("weights/dataset_cache")
    ckpt_dir = Path("weights/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_2d_weight_file = ckpt_dir / "best_bst_2d_global.pt"
    resume_path = ckpt_dir / "latest_2d_checkpoint.pt"

    print(f"\n🚀 Starting 2D Training Pipeline on Device: {device}")

    # 1. Load Pre-Split Cached Data
    if not (cache_dir / "train_data.pkl").exists():
        print("❌ Error: Cached data not found. Run the 08_prepare_global_dataset.py script first.")
        return

    train_loader = DataLoader(Badminton2DDataset(cache_dir/"train_data.pkl"), batch_size=64, shuffle=True)
    val_loader   = DataLoader(Badminton2DDataset(cache_dir/"val_data.pkl"), batch_size=64)
    test_loader  = DataLoader(Badminton2DDataset(cache_dir/"test_data.pkl"), batch_size=64)

    # 2. Initialize 2D Model
    # in_dim=72 (17 keypoints + 19 bones) * 2 channels
    model = BST_CG_AP(in_dim=72, seq_len=100, n_class=35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    total_epochs = 150
    start_epoch = 1
    best_val_acc = 0.0
    best_weights = None
    early_stop_patience = 25
    early_stop_counter = 0

    # 3. Resume Logic
    if resume_path.exists():
        print(f"🔄 Resuming 2D training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=400, num_training_steps=total_epochs * len(train_loader))

    # 4. Training Loop
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"2D Ep {epoch:03d}/{total_epochs}", leave=False)
        
        for (j_batch, p_batch, s_batch), vlen, l_batch in pbar:
            optimizer.zero_grad()
            output = model(j_batch.to(device), s_batch.to(device), p_batch.to(device), vlen.to(device))
            loss = criterion(output, l_batch.to(device))
            loss.backward(); optimizer.step(); scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        # Validation
        metrics = evaluate_comprehensive(model, val_loader, device)
        
        if metrics['acc'] > best_val_acc:
            best_val_acc = metrics['acc']
            best_weights = deepcopy(model.state_dict())
            early_stop_counter = 0
            # Save Best Weights immediately
            torch.save({'model_state_dict': best_weights, 'metrics': metrics}, best_2d_weight_file)
            print(f"⭐ [New Best] Acc: {metrics['acc']:.4f} | Acc@2: {metrics['acc2']:.4f} | mF1: {metrics['macro_f1']:.4f}")
        else:
            early_stop_counter += 1
            
        print(f"📊 [Val] Ep {epoch:03d} | Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Acc: {metrics['acc']:.4f} | mF1: {metrics['macro_f1']:.4f} | Patience: {early_stop_counter}/{early_stop_patience}")

        # Periodic Checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_acc': best_val_acc
            }, resume_path)

        if early_stop_counter >= early_stop_patience:
            print(f"\n🛑 Early Stopping triggered at epoch {epoch}.")
            break

    # 5. Final Global Test
    print(f"\n{'='*70}\n🏁 TRAINING COMPLETE. LOADING BEST WEIGHTS FOR FINAL TEST...")
    if best_weights:
        model.load_state_dict(best_weights)
    
    test_metrics = evaluate_comprehensive(model, test_loader, device)
    print(f"🏆 FINAL 2D TEST RESULTS")
    print(f"  Accuracy (Top-1) : {test_metrics['acc']*100:.2f}%")
    print(f"  Accuracy (Top-2) : {test_metrics['acc2']*100:.2f}%")
    print(f"  Macro-F1 Score   : {test_metrics['macro_f1']:.4f}")
    print(f"  Min-F1 Score     : {test_metrics['min_f1']:.4f}\n{'='*70}")

if __name__ == "__main__":
    main()