import os
import cv2
import pickle
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
from tqdm import tqdm

# Standard imports
from pose_estimation import load_pose_backend, estimate_poses_batched
from court_calibration import load_calibration, project_to_pixel
from config import WORLD_PTS

def main():
    # --- CONFIGURATION ---
    NUM_RALLIES_TO_TEST = 5
    VIDEO_NAME = "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals"
    BATCH_SIZE = 128
    
    # Load the module dynamically to avoid the leading-digit import error
    spec = importlib.util.spec_from_file_location("pose_module", "03_estimate_pose.py")
    pose_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pose_module)
    
    asset_dir = Path("test_assets")
    video_path = asset_dir / f"{VIDEO_NAME}.mp4"
    base_out = Path("test_shuttlenet") / VIDEO_NAME
    calib_dir = base_out / "calib_out"
    test_out_dir = base_out / "pose_test_subset"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    set_dir = Path("shuttleset/set") / VIDEO_NAME
    dfs = []
    for f in os.listdir(set_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(set_dir / f)
            df['set_file'] = f
            dfs.append(df)
    
    gt_df = pd.concat(dfs).dropna(subset=["frame_num"])
    gt_df["frame_num"] = gt_df["frame_num"].astype(int)
    
    grouped = gt_df.sort_values(by=["set_file", "rally", "frame_num"]).groupby(["set_file", "rally"])
    rally_keys = list(grouped.groups.keys())[:NUM_RALLIES_TO_TEST]

    # 2. Extract frames
    cap_orig = cv2.VideoCapture(str(video_path))
    n_total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    
    test_frames = []
    first_hit_indices = set()
    current_idx = 0

    print(f"--- Extracting frames for {NUM_RALLIES_TO_TEST} rallies ---")
    for key in tqdm(rally_keys, desc="Extracting Rallies"):
        group = grouped.get_group(key)
        first_hit, last_hit = int(group["frame_num"].min()), int(group["frame_num"].max())
        start_f, end_f = max(0, first_hit - 60), min(n_total, last_hit + 61)
        
        first_hit_indices.add(current_idx + (first_hit - start_f))
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(start_f, end_f):
            ret, frame = cap_orig.read()
            if not ret: break
            test_frames.append(frame)
            current_idx += 1
    cap_orig.release()

    # 3. Models and Calibration
    P, K, rvec, tvec = load_calibration(calib_dir)
    court_poly = project_to_pixel(WORLD_PTS[:4], P).astype(np.int32)
    det_m, pose_m, _ = load_pose_backend()

    # 4. Run Batch Estimation with tqdm
    print(f"--- Running Estimation on {len(test_frames)} frames ---")
    results = []
    # Manual batching to show progress bar
    for i in tqdm(range(0, len(test_frames), BATCH_SIZE), desc="Processing Batches"):
        batch = test_frames[i : i + BATCH_SIZE]
        batch_res = estimate_poses_batched(
            batch, 
            court_poly, 
            K, rvec, tvec, 
            det_m, pose_m, 
            batch_size=len(batch),
            first_hit_indices=first_hit_indices,
            start_idx=i
        )
        results.extend(batch_res)

    # 5. Render Video with fixed drawing function
    vid_out = test_out_dir / "subset_debug.mp4"
    h, w = test_frames[0].shape[:2]
    out_vid = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    print("--- Rendering debug video ---")
    for i in tqdm(range(len(test_frames)), desc="Rendering"):
        frame = test_frames[i].copy()
        # Passing results[i] to the module's draw function
        pose_module.custom_pose_draw(frame, results[i])
        if i in first_hit_indices:
            cv2.putText(frame, "STATE RESET", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        out_vid.write(frame)
    
    out_vid.release()
    print(f"Done! Check: {vid_out}")

if __name__ == "__main__":
    main()