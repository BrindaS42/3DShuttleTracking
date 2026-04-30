import os
import sys
import pickle
import numpy as np
import pandas as pd
import random
from pathlib import Path
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
    """Extracts 2D+3D shuttle and pose features into 100-frame sequences."""
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
        
        shuttle_3d_seq[t_idx] = np.nan_to_num(traj_3d_raw[trim_idx]) / [COURT_W, COURT_L, COURT_H]
        s2d = np.nan_to_num(shuttle_2d_raw[trim_idx])
        shuttle_2d_seq[t_idx] = np.clip(s2d / [NORM_W, NORM_H], 0, 1)

        pf = poses[trim_idx]
        for p_idx, p_attr in enumerate(['near', 'far']):
            player = getattr(pf, p_attr, None)
            if player and player.keypoints:
                kpts_raw = np.array(player.keypoints)[:, :2]
                max_x = np.max(kpts_raw[:, 0])
                cur_w, cur_h = (1280.0, 720.0) if max_x <= 1280 else (1920.0, 1080.0)
                
                kpts = np.clip(kpts_raw / [cur_w, cur_h], 0, 1)
                bones = np.zeros((19, 2))
                for b_idx, (j1, j2) in enumerate(COCO_BONES):
                    bones[b_idx] = kpts[j1] - kpts[j2]
                jnb_seq[t_idx, p_idx, :34] = kpts.flatten()
                jnb_seq[t_idx, p_idx, 34:72] = bones[:19].flatten()
                if player.floor_pos_3d is not None:
                    pos_seq[t_idx, p_idx] = np.nan_to_num(player.floor_pos_3d[:2]) / [COURT_W, COURT_L]
                    
    return jnb_seq, pos_seq, shuttle_2d_seq, shuttle_3d_seq

def process_match_set(match_list, root_dir, gt_dir, desc):
    """Aggregates all samples for a specific set of matches."""
    S_jnb, S_pos, S_s2d, S_s3d, S_labels = [], [], [], [], []
    
    for match_name in tqdm(match_list, desc=desc):
        match_folder = root_dir / match_name
        traj_path = match_folder / "traj_out" / f"{match_name}_traj_3d_trimmed.npy"
        s2d_path = match_folder / "shuttle_out" / "shuttle_trimmed.npy"
        pose_path = match_folder / "pose_out" / "poses.pkl"
        csv_dir = gt_dir / match_name
        
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
                    S_jnb.append(jnb); S_pos.append(pos); S_s2d.append(s2d); S_s3d.append(s3d); S_labels.append(ALL_TYPES.index(full_label))
    
    return {'jnb': S_jnb, 'pos': S_pos, 's2d': S_s2d, 's3d': S_s3d, 'labels': S_labels}

def main():
    root_dir, gt_dir = Path("test_shuttlenet"), Path("shuttleset/set")
    out_dir = Path("weights/dataset_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_matches = sorted([f.name for f in root_dir.iterdir() if f.is_dir()])
    valid_matches, discarded = [], []

    # 1. Validation & Discard Logic
    for m in all_matches:
        m_path = root_dir / m
        missing = []
        if not (m_path / "traj_out" / f"{m}_traj_3d_trimmed.npy").exists(): missing.append("3D Traj")
        if not (m_path / "shuttle_out" / "shuttle_trimmed.npy").exists(): missing.append("2D Shuttle")
        if not (m_path / "pose_out" / "poses.pkl").exists(): missing.append("Poses")
        if not (gt_dir / m).exists(): missing.append("GroundTruth CSV")
        
        if missing:
            discarded.append(f"{m:20} | Reason: Missing {', '.join(missing)}")
        else:
            valid_matches.append(m)

    # 2. Match-wise Split (80% Train, 10% Val, 10% Test)
    random.seed(42)
    random.shuffle(valid_matches)
    
    train_m, temp_m = train_test_split(valid_matches, test_size=0.20, random_state=42)
    val_m, test_m = train_test_split(temp_m, test_size=0.50, random_state=42)

    # 3. Logging
    print("\n" + "="*50)
    print(f"📊 MATCH-WISE SPLIT LOG (Total Videos: {len(all_matches)})")
    print("="*50)
    print(f"✅ Valid Matches   : {len(valid_matches)}")
    print(f"❌ Discarded       : {len(discarded)}")
    for d in discarded: print(f"   - {d}")
    print("-" * 50)
    print(f"📁 TRAIN SET ({len(train_m)}): {', '.join(train_m[:3])}...")
    print(f"📁 VAL SET   ({len(val_m)}): {', '.join(val_m)}")
    print(f"📁 TEST SET  ({len(test_m)}): {', '.join(test_m)}")
    print("="*50 + "\n")

    # 4. Processing & Saving
    for name, m_list in [("train", train_m), ("val", val_m), ("test", test_m)]:
        data = process_match_set(m_list, root_dir, gt_dir, f"Processing {name} set")
        with open(out_dir / f"{name}_data.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Saved {name}_data.pkl ({len(data['labels'])} instances)")

if __name__ == "__main__": main()