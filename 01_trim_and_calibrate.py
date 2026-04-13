import os
import sys
import logging
import subprocess
from pathlib import Path
import cv2
import pandas as pd

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        console = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
    return logger

def get_calib_frame(gt_df):
    for rally_id, group in gt_df.groupby("rally"):
        if len(group) > 1:
            return int(group.iloc[1]["frame_num"]) # 2nd shot
    return int(gt_df.iloc[0]["frame_num"]) if not gt_df.empty else 0

def main():
    asset_dir = Path("test_assets")
    if not asset_dir.exists():
        print("[!] test_assets folder not found.")
        return

    for video_path in asset_dir.glob("*.mp4"):
        match_folder = video_path.stem
        base_out = Path("test_shuttlenet") / match_folder
        base_out.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logger(f"TrimCalib_{match_folder}", base_out / "trim_calib.log")
        logger.info(f"=== Processing Video: {match_folder} ===")

        # Load and sort Ground Truth
        set_dir = Path("shuttleset/set") / match_folder
        if not set_dir.exists():
            logger.error(f"GT folder {set_dir} not found. Skipping.")
            continue

        dfs = []
        for set_file in os.listdir(set_dir):
            if set_file.startswith("set") and set_file.endswith(".csv"):
                df = pd.read_csv(set_dir / set_file, encoding='utf-8')
                if 'frame_nur' in df.columns: df.rename(columns={'frame_nur': 'frame_num'}, inplace=True)
                df['set_file'] = set_file
                dfs.append(df)
        
        if not dfs:
            logger.error("No CSVs found. Skipping.")
            continue
            
        gt_df = pd.concat(dfs, ignore_index=True).dropna(subset=["frame_num"])
        gt_df["frame_num"] = gt_df["frame_num"].astype(int)
        # Sort by rally then shot number (frame_num)
        gt_df = gt_df.sort_values(by=["rally", "frame_num"]).reset_index(drop=True)

        calib_target_frame = get_calib_frame(gt_df)
        logger.info(f"Selected frame {calib_target_frame} for calibration (2nd shot of a rally).")


        cap = cv2.VideoCapture(str(video_path))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        trimmed_video_path = base_out / "trimmed_temp.mp4"
        
        required_frames = set()
        for (set_file, rally_id), group in gt_df.groupby(["set_file", "rally"]):
            num_shots = len(group)
            first_hit, last_hit = int(group["frame_num"].min()), int(group["frame_num"].max())            
            logger.info(f"Queuing for slice -> Set: {set_file} | Rally: {rally_id} | Total Shots: {num_shots} | Hit Frames: {first_hit} to {last_hit}")
            for f in range(max(0, first_hit - 60), min(n_total, last_hit + 61)): 
                required_frames.add(f)
                
        fast_lookup = set(sorted(list(required_frames)))
        max_frame = max(fast_lookup) if fast_lookup else 0
        
        if not trimmed_video_path.exists():
            logger.info(f"Slicing video to {trimmed_video_path}...")
            out = cv2.VideoWriter(str(trimmed_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            curr_frame = 0
            while curr_frame <= max_frame:
                if curr_frame in fast_lookup:
                    ret, frame = cap.read()
                    if ret: out.write(frame)
                else:
                    ret = cap.grab()
                if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 1)
                curr_frame += 1
            out.release()
        else:
            logger.info("Trimmed video already exists. Skipping trim.")

        # Calibration
        calib_dir = base_out / "calib_out"
        calib_dir.mkdir(parents=True, exist_ok=True)
        if not (calib_dir / "P.npy").exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, calib_target_frame)
            ret, frame = cap.read()
            if ret:
                img_path = str(calib_dir / "calib_frame.jpg")
                cv2.imwrite(img_path, frame)
                logger.info("Launching manual calibration...")
                subprocess.run([sys.executable, "court_calibration.py", "--image", img_path, "--out_dir", str(calib_dir)])
            else:
                logger.error(f"Could not read frame {calib_target_frame} for calibration.")
        else:
            logger.info("Calibration P.npy already exists. Skipping.")
        cap.release()

if __name__ == "__main__": main()