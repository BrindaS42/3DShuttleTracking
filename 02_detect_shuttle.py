import os
import sys
import logging
from pathlib import Path
import numpy as np

from config import TRACKNET_DIR
from shuttle_detection import run_tracknet, parse_tracknet_csv
from shuttle_detection import render_debug_video as dump_shuttle_video

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

def main():
    asset_dir = Path("test_assets")
    if not asset_dir.exists():
        print("[!] test_assets folder not found.")
        return

    for video_path in asset_dir.glob("*.mp4"):
        match_folder = video_path.stem
        base_out = Path("test_shuttlenet") / match_folder
        trimmed_video_path = base_out / "trimmed_temp.mp4"
        shuttle_dir = base_out / "shuttle_out"
        shuttle_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(f"Shuttle_{match_folder}", shuttle_dir / "shuttle_det.log")
        logger.info(f"=== Processing Shuttle Detection: {match_folder} ===")

        if not trimmed_video_path.exists():
            logger.error(f"Trimmed video missing for {match_folder}. Run Step 1 first.")
            continue

        shuttle_cache = shuttle_dir / "shuttle_trimmed.npy"
        if shuttle_cache.exists():
            logger.info("Raw shuttle detections already exist. Skipping inference.")
            trimmed_shuttle = np.load(shuttle_cache)
        else:
            logger.info("Running TrackNetV3 on trimmed video...")
            csv_path = run_tracknet(video_path=str(trimmed_video_path), weights_dir=str(TRACKNET_DIR), raw_out_dir=str(shuttle_dir), eval_mode="weight", batch_size=4, large_video=True)
            trimmed_shuttle = parse_tracknet_csv(csv_path, str(trimmed_video_path))
            # DO NOT clean or fill gaps here. Save raw.
            np.save(shuttle_cache, trimmed_shuttle)
            logger.info(f"Saved RAW detections to {shuttle_cache}")

        vid_out = shuttle_dir / f"{match_folder}_shuttle_trimmed.mp4"
        if not vid_out.exists():
            logger.info("Dumping raw annotated shuttle video...")
            dump_shuttle_video(str(trimmed_video_path), trimmed_shuttle, str(vid_out))
        else:
            logger.info("Annotated video already exists.")

if __name__ == "__main__": main()