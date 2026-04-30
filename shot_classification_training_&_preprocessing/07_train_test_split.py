import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "stroke_classification"))
from preparing_data.shuttleset_dataset import get_stroke_types

# Constants
NORM_W, NORM_H = 640.0, 360.0 
COURT_W, COURT_L, COURT_H = 6.1, 13.4, 5.0 
COCO_BONES = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
              (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
ALL_TYPES = get_stroke_types()

def extract_features_dual(poses, start_f, end_f, hit_n, shuttle_2d_raw, traj_3d_raw, trim_lookup):
    """Extracts features for both 2D and 3D shuttle inputs simultaneously."""
    jnb_seq = np.zeros((100, 2, 72), dtype=np.float32)
    pos_seq = np.zeros((100, 2, 2), dtype=np.float32)
    shuttle_2d_seq = np.zeros((100, 2), dtype=np.float32)
    shuttle_3d_seq = np.zeros((100, 3), dtype=np.float32)
    seq_start = 50 + (start_f - hit_n)
    
    for i, orig_f in enumerate(range(start_f, end_f + 1)):
        t_idx = seq_start + i
        if t_idx < 0 or t_idx >= 100: continue
        trim_idx = trim_lookup.get(orig_f, -1)
        if trim_idx == -1 or trim_idx >= len(poses): continue
        
        # Shuttle 3D (World Coords)
        shuttle_3d_seq[t_idx] = np.nan_to_num(traj_3d_raw[trim_idx]) / [COURT_W, COURT_L, COURT_H]
        # Shuttle 2D (Pixel Coords with Auto-Res Detect)
        s2d = np.nan_to_num(shuttle_2d_raw[trim_idx])
        shuttle_2d_seq[t_idx] = np.clip(s2d / [NORM_W, NORM_H], 0, 1)

        pf = poses[trim_idx]
        for p_idx, p_attr in enumerate(['near', 'far']):
            player = getattr(pf, p_attr, None)
            if player and player.keypoints:
                kpts_raw = np.array(player.keypoints)[:, :2]
                max_x = np.max(kpts_raw[:, 0])
                cur_w = 1280.0 if max_x <= 1280 else 1920.0 # Basic auto-detect
                cur_h = 720.0 if max_x <= 1280 else 1080.0
                
                kpts = np.clip(kpts_raw / [cur_w, cur_h], 0, 1)
                bones = np.zeros((19, 2))
                for b_idx, (j1, j2) in enumerate(COCO_BONES):
                    bones[b_idx] = kpts[j1] - kpts[j2]
                jnb_seq[t_idx, p_idx, :34] = kpts.flatten()
                jnb_seq[t_idx, p_idx, 34:72] = bones[:19].flatten()
                if player.floor_pos_3d is not None:
                    pos_seq[t_idx, p_idx] = np.nan_to_num(player.floor_pos_3d[:2]) / [COURT_W, COURT_L]
                    
    return jnb_seq, pos_seq, shuttle_2d_seq, shuttle_3d_seq

def main():
    root_dir, gt_dir = Path("test_shuttlenet"), Path("shuttleset/set")
    out_dir = Path("weights/dataset_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    G_jnb, G_pos, G_s2d, G_s3d, G_labels = [], [], [], [], []

    match_folders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
    for match_folder in tqdm(match_folders, desc="Aggregating Global Data"):
        match_name = match_folder.name
        traj_path = match_folder / "traj_out" / f"{match_name}_traj_3d_trimmed.npy"
        s2d_path = match_folder / "shuttle_out" / "shuttle_trimmed.npy"
        pose_path = match_folder / "pose_out" / "poses.pkl"
        csv_dir = gt_dir / match_name
        
        if not (traj_path.exists() and s2d_path.exists() and pose_path.exists() and csv_dir.exists()):
            continue
        
        traj_3d_raw = np.load(traj_path)
        shuttle_2d_raw = np.load(s2d_path)
        with open(pose_path, "rb") as f: poses = pickle.load(f)
        dfs = [pd.read_csv(f).assign(set_file=f.name) for f in csv_dir.glob("*.csv")]
        gt_df = pd.concat(dfs).dropna(subset=["frame_num"])
        gt_df["frame_num"] = gt_df["frame_num"].astype(int)

        required_frames = []
        for _, rally_group in gt_df.groupby(["set_file", "rally"]):
            f_start, f_end = int(rally_group["frame_num"].min()), int(rally_group["frame_num"].max())
            required_frames.extend(range(max(0, f_start-60), min(1000000, f_end+61)))
        trim_lookup = {orig_f: idx for idx, orig_f in enumerate(required_frames)}

        for _, rally_group in gt_df.groupby("rally"):
            hits = rally_group.sort_values("frame_num").to_dict('records')
            for i in range(len(hits)):
                hit_n = int(hits[i]["frame_num"])
                py, oy = hits[i].get('player_location_y', 0), hits[i].get('opponent_location_y', 0)
                full_label = f"{'Top' if py < oy else 'Bottom'}_{hits[i].get('type', hits[i].get('stroke_type', ''))}"
                
                if full_label in ALL_TYPES:
                    start_f = max(0, max(int(hits[i-1]["frame_num"]) if i > 0 else 0, hit_n - 50))
                    end_f = min(1000000, min(int(hits[i+1]["frame_num"]) if i < len(hits)-1 else 1000000, hit_n + 49))
                    jnb, pos, s2d, s3d = extract_features_dual(poses, start_f, end_f, hit_n, shuttle_2d_raw, traj_3d_raw, trim_lookup)
                    G_jnb.append(jnb); G_pos.append(pos); G_s2d.append(s2d); G_s3d.append(s3d); G_labels.append(ALL_TYPES.index(full_label))

    # SPLITTING
    indices = np.arange(len(G_labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, stratify=G_labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=[G_labels[i] for i in temp_idx], random_state=42)

    def save_split(name, idx_list):
        data = {
            'jnb': [G_jnb[i] for i in idx_list], 'pos': [G_pos[i] for i in idx_list],
            's2d': [G_s2d[i] for i in idx_list], 's3d': [G_s3d[i] for i in idx_list],
            'labels': [G_labels[i] for i in idx_list]
        }
        with open(out_dir / f"{name}_data.pkl", "wb") as f: pickle.dump(data, f)
        print(f"✅ Saved {name} set ({len(idx_list)} samples)")

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)

if __name__ == "__main__": main()