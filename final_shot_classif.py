"""
final_shot_classif.py
═════════════════════
Interactive end-to-end evaluation script mapping the ShuttleSet directory structure
to the BST model pipeline. 
"""

import os
import sys
import logging
import pickle
import ast
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from config import TRACKNET_DIR
from court_calibration import load_calibration
from shuttle_detection import run_tracknet, parse_tracknet_csv, clean_detections
from shuttle_detection import render_debug_video as dump_shuttle_video
from pose_estimation import load_pose_backend, estimate_poses_batched
from shot_classification import (
    normalize_rally_data, fill_linear_shuttle_gaps, build_court_homography
)

BST_ROOT = Path(__file__).parent / "stroke_classification"
if str(BST_ROOT.resolve()) not in sys.path:
    sys.path.insert(0, str(BST_ROOT.resolve()))

from stroke_classification.preparing_data.shuttleset_dataset import (
    get_merged_stroke_types, get_bone_pairs, create_bones
)
from stroke_classification.model.bst import BST_CG_AP

SHUTTLESET_18_TO_BST_12 = {
    "發短球": "發短球", "發長球": "發長球", "長球": "長球",
    "殺球": "殺球", "點扣": "殺球", 
    "切球": "切球", "過渡切球": "切球",
    "挑球": "挑球", "防守回挑": "挑球",
    "平球": "平球", "小平球": "平球", "後場抽平球": "平球", "防守回抽": "平球",
    "放小球": "放小球", "擋小球": "擋小球", "勾球": "勾球",
    "推球": "推球", "撲球": "撲球",
    "未知球種": "Unknown", "Unknown": "Unknown"
}

BST_TO_ENGLISH = {
    "發短球": "Short Service", "發長球": "Long Service", "長球": "Clear",
    "殺球": "Smash", "切球": "Drop", "挑球": "Lob",
    "平球": "Drive", "放小球": "Net shot", "擋小球": "Return net", 
    "勾球": "Cross-court net", "推球": "Push", "撲球": "Rush", 
    "未知球種": "Unknown", "Unknown": "Unknown"
}

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(out_dir / "eval.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("ShuttleSetEval")

def compute_metrics(results_df: pd.DataFrame, out_dir: Path, logger: logging.Logger):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
    eval_df = results_df[(results_df["gt_type"] != "接不到") & (results_df["bst_pred_stroke"] != "SKIP")].copy()
    if eval_df.empty:
        logger.warning("No valid strokes to evaluate for accuracy.")
        return

    stroke_acc = accuracy_score(eval_df["gt_mapped_stroke"], eval_df["bst_pred_stroke"])
    precision  = precision_score(eval_df["gt_mapped_stroke"], eval_df["bst_pred_stroke"], average='weighted', zero_division=0)
    recall     = recall_score(eval_df["gt_mapped_stroke"], eval_df["bst_pred_stroke"], average='weighted', zero_division=0)
    f1         = f1_score(eval_df["gt_mapped_stroke"], eval_df["bst_pred_stroke"], average='weighted', zero_division=0)

    side_eval_df = eval_df[eval_df["gt_side"].isin(["Top", "Bottom"])]
    side_acc = accuracy_score(side_eval_df["gt_side"], side_eval_df["bst_pred_side"]) if not side_eval_df.empty else 0.0
    exact_match = ((side_eval_df["gt_mapped_stroke"] == side_eval_df["bst_pred_stroke"]) & 
                   (side_eval_df["gt_side"] == side_eval_df["bst_pred_side"])).mean() if not side_eval_df.empty else 0.0

    logger.info("\n" + "═"*50)
    logger.info(" FINAL ACCURACY METRICS")
    logger.info("═"*50)
    logger.info(f" Total Strokes Evaluated : {len(eval_df)}")
    logger.info(f" Stroke Accuracy         : {stroke_acc:.4f}")
    logger.info(f" Stroke Precision (W-Avg): {precision:.4f}")
    logger.info(f" Stroke Recall    (W-Avg): {recall:.4f}")
    logger.info(f" Stroke F1-Score  (W-Avg): {f1:.4f}")
    logger.info(f" Side (Top/Bottom) Acc   : {side_acc:.4f}")
    logger.info(f" Exact Match Accuracy    : {exact_match:.4f}")
    logger.info("═"*50)

    report = classification_report(eval_df["gt_mapped_stroke"], eval_df["bst_pred_stroke"], zero_division=0)
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("═"*50 + "\nDETAILED CLASSIFICATION REPORT\n" + "═"*50 + "\n\n")
        f.write(report)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    labels = sorted(eval_df["gt_mapped_stroke"].unique())
    cm = confusion_matrix(eval_df["gt_mapped_stroke"], eval_df["bst_pred_stroke"], labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Ground Truth (Mapped)')
    plt.xlabel('Predicted by BST')
    plt.title(f'Stroke Confusion Matrix (Accuracy: {stroke_acc:.2f} | F1: {f1:.2f})')
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
def create_rally_subset_video(video_path: str, gt_df: pd.DataFrame, n_total: int, out_path: str, logger: logging.Logger, buffer: int = 60, skip_extraction: bool = False):
    logger.info("Calculating required frames for selected rallies...")
    required_frames = set()
    for rally_id, group in gt_df.groupby(["set_file", "rally"]):
        first_hit = int(group["frame_num"].min())
        last_hit = int(group["frame_num"].max())
        for f in range(max(0, first_hit - buffer), min(n_total, last_hit + buffer + 1)):
            required_frames.add(f)
            
    sorted_frames = sorted(list(required_frames))
    if skip_extraction:
        logger.info(f"Skipping video slicing. Mapped {len(sorted_frames)} frames instantly.")
        return sorted_frames
    
    chunks = []
    if sorted_frames:
        start, last = sorted_frames[0], sorted_frames[0]
        for f in sorted_frames[1:]:
            if f == last + 1: last = f
            else:
                chunks.append((start, last))
                start, last = f, f
        chunks.append((start, last))
        
    logger.info("Slicing video (Fast Chunk-Seek Mode)...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for i, (start_f, end_f) in enumerate(chunks):
        seek_target = max(0, start_f - 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
        curr_frame = seek_target
        while curr_frame < start_f:
            cap.read()
            curr_frame += 1
            
        bad_consecutive = 0
        while curr_frame <= end_f:
            ret, frame = cap.read()
            if not ret:
                bad_consecutive += 1
                if bad_consecutive > 5:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 1)
                    bad_consecutive = 0
            else:
                bad_consecutive = 0
                out.write(frame)
            curr_frame += 1
            
    cap.release()
    out.release()
    logger.info(f"Temporary trimmed video created at {out_path}")
    return sorted_frames

def run_manual_calibration(video_path: str, calib_dir: str, frame_idx: int, logger: logging.Logger):
    logger.info(f"Extracting frame {frame_idx} for calibration...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: return
        
    img_path = os.path.join(calib_dir, "calib_frame.jpg")
    cv2.imwrite(img_path, frame)
    cmd = [sys.executable, "court_calibration.py", "--image", img_path, "--out_dir", calib_dir]
    subprocess.run(cmd)

class EmptyPoseFrame:
    near = None
    far = None

def draw_pose_frame(frame, pf_obj, coco_pairs, draw_coords=False, H_court=None):
    if not pf_obj: return
    players = []
    if hasattr(pf_obj, 'near') and pf_obj.near is not None: players.append(("Near", pf_obj.near))
    if hasattr(pf_obj, 'far') and pf_obj.far is not None: players.append(("Far", pf_obj.far))

    for label, person in players:
        kpts = None
        if hasattr(person, 'keypoints') and person.keypoints is not None: kpts = np.array(person.keypoints)

        if kpts is not None and kpts.ndim >= 2 and kpts.shape[0] >= 17:
            for p1, p2 in coco_pairs:
                x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
                x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0: cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x, y in kpts[:, :2]:
                if x > 0 and y > 0: cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

            if draw_coords and H_court is not None:
                ankles = [px for px in [person.left_ankle_px, person.right_ankle_px] if px is not None]
                if ankles:
                    u_avg, v_avg = np.mean(ankles, axis=0)
                    p_h = H_court @ np.array([u_avg, v_avg, 1.0], dtype=np.float64)
                    court_x, court_y = p_h[:2] / p_h[2]
                    
                    valid_kpts = [k for k in kpts if k[0] > 0 and k[1] > 0]
                    if valid_kpts:
                        min_y = min([k[1] for k in valid_kpts])
                        avg_x = sum([k[0] for k in valid_kpts]) / len(valid_kpts)
                        text = f"{label}: (X:{court_x:.1f}, Y:{court_y:.1f})"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (int(avg_x)-35, int(min_y)-th-15), (int(avg_x)-35+tw, int(min_y)-5), (0,0,0), -1)
                        cv2.putText(frame, text, (int(avg_x)-30, int(min_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def dump_pose_video(video_path: str, trimmed_poses: list, out_path: str, H_court=None):
    print("Generating Pose Annotated Video (with Coordinates)...")
    cap = cv2.VideoCapture(video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    coco_pairs = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx < len(trimmed_poses):
            draw_pose_frame(frame, trimmed_poses[frame_idx], coco_pairs, draw_coords=True, H_court=H_court)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def dump_final_annotated_video(trimmed_video_path: str, out_path: Path, trimmed_shuttle: np.ndarray, trimmed_poses: list, results_df: pd.DataFrame, frame_map: list, logger: logging.Logger, H_court=None):
    logger.info("Generating Final BST Annotated Video...")
    cap = cv2.VideoCapture(trimmed_video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    pred_dict = {
        row["frame_num"]: {
            "p_side": row["bst_pred_side"], "p_stroke": row["bst_pred_stroke"],
            "gt_side": row["gt_side"], "gt_stroke": row["gt_mapped_stroke"]
        } for _, row in results_df.iterrows() if row["bst_pred_stroke"] != "SKIP"
    }

    coco_pairs = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
    frame_idx, text_timer = 0, 0
    text_pred, text_gt = "", ""
    color_pred = (255, 255, 255)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        orig_f = frame_map[frame_idx] if frame_idx < len(frame_map) else -1

        if frame_idx < len(trimmed_shuttle) and not np.isnan(trimmed_shuttle[frame_idx][0]):
            cv2.circle(frame, (int(trimmed_shuttle[frame_idx][0]), int(trimmed_shuttle[frame_idx][1])), 6, (0, 0, 255), -1)
                
        if frame_idx < len(trimmed_poses):
            draw_pose_frame(frame, trimmed_poses[frame_idx], coco_pairs, draw_coords=False, H_court=H_court)
                            
        if orig_f in pred_dict:
            d = pred_dict[orig_f]
            p_en, gt_en = BST_TO_ENGLISH.get(d["p_stroke"], d["p_stroke"]), BST_TO_ENGLISH.get(d["gt_stroke"], d["gt_stroke"])
            text_pred, text_gt = f"Pred: {d['p_side']} | {p_en}", f"GT  : {d['gt_side']} | {gt_en}"
            color_pred = (0, 255, 0) if (d["p_side"] == d["gt_side"] and d["p_stroke"] == d["gt_stroke"]) else (0, 0, 255)
            text_timer = int(fps * 1.5) 
            
        if text_timer > 0:
            (tw1, th1), _ = cv2.getTextSize(text_pred, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (35, 35), (45 + tw1, 55 + th1), (0, 0, 0), -1)
            cv2.putText(frame, text_pred, (40, 40 + th1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_pred, 2)
            
            (tw2, th2), _ = cv2.getTextSize(text_gt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (35, 65 + th1), (45 + tw2, 85 + th1 + th2), (0, 0, 0), -1)
            cv2.putText(frame, text_gt, (40, 70 + th1 + th2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            text_timer -= 1
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()

def load_shuttleset_gt(match_folder: str, scope: dict) -> pd.DataFrame:
    base_dir = Path("shuttleset/set") / match_folder
    dfs = []
    for set_file in scope["sets"]:
        df = pd.read_csv(base_dir / set_file, encoding='utf-8') 
        if 'frame_nur' in df.columns: df.rename(columns={'frame_nur': 'frame_num'}, inplace=True)
        df['set_file'] = set_file
        if scope["rallies"]: df = df[df['rally'].isin(scope["rallies"])]
        dfs.append(df)
        
    if not dfs: return pd.DataFrame()
    combined_df = pd.concat(dfs, ignore_index=True).dropna(subset=["frame_num"])
    combined_df["frame_num"] = combined_df["frame_num"].astype(int)
    
    if 'player_location_y' in combined_df.columns and 'opponent_location_y' in combined_df.columns:
        valid_pos_mask = combined_df['player_location_y'].notna() & combined_df['opponent_location_y'].notna()
        combined_df.loc[valid_pos_mask, 'hitter_side_gt'] = np.where(
            combined_df.loc[valid_pos_mask, 'player_location_y'] < combined_df.loc[valid_pos_mask, 'opponent_location_y'],
            'Top', 'Bottom'
        )
    else: combined_df['hitter_side_gt'] = 'Unknown'

    combined_df["gt_type"] = combined_df["type"].astype(str)
    combined_df["skip"] = combined_df["gt_type"] == "接不到"
    return combined_df.sort_values("frame_num").reset_index(drop=True)

def main():
    print("🏸 ShuttleSet Interactive Evaluation Pipeline 🏸")
    match_folder = input("\nEnter the match folder name (e.g., Kento_MOMOTA...): ").strip()
    video_path = Path(f"test_assets/{match_folder}.mp4")
    if not video_path.exists(): sys.exit(1)

    base_out = Path("test_shuttlenet") / match_folder
    shuttle_dir, pose_dir, shot_dir = base_out/"shuttle_out", base_out/"pose_out", base_out/"shot_out"
    for d in [shuttle_dir, pose_dir, shot_dir]: d.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(shot_dir)

    print("\n═ RUN SCOPE CONFIGURATION ═")
    avail_sets = sorted([f for f in os.listdir(Path("shuttleset/set") / match_folder) if f.startswith("set")])
    print("1. Process specific Set No. AND exact Rally Numbers\n2. Process specific Set No. (entire set)\n3. Process ALL sets")
    c = input("Select an option (1/2/3): ").strip()
    
    scope = {"sets": avail_sets, "rallies": None}
    if c in ['1', '2']:
        set_choice = input("Enter Set filename: ").strip()
        scope["sets"] = [set_choice] if set_choice in avail_sets else avail_sets
        if c == '1': scope["rallies"] = [int(r.strip()) for r in input("Enter Rally numbers (comma separated): ").split(',')]

    gt_df = load_shuttleset_gt(match_folder, scope)
    if gt_df.empty: sys.exit(1)
    
    first_rally_num = gt_df["rally"].iloc[0]
    first_rally_df = gt_df[gt_df["rally"] == first_rally_num]
    
    first_hit_frame = int(first_rally_df["frame_num"].min())
    calib_target_frame = int(first_rally_df.iloc[1]["frame_num"]) if len(first_rally_df) > 1 else first_hit_frame

    cap = cv2.VideoCapture(str(video_path))
    v_w, v_h, n_total = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    trimmed_video_path = str(base_out / "trimmed_temp.mp4")
    skip_trimming = False
    if Path(trimmed_video_path).exists():
        if input(f"\n[?] Found existing trimmed video at {trimmed_video_path}.\nSkip extraction? (y/n): ").strip().lower() == 'y':
            skip_trimming = True

    frame_map = create_rally_subset_video(str(video_path), gt_df, n_total, trimmed_video_path, logger, skip_extraction=skip_trimming)
    n_trimmed = len(frame_map)

    print("\n═ HOMOGRAPHY CONFIGURATION ═")
    print("1. Use existing stored (cache in calib_out)\n2. Parse from shuttleset/set/homography.csv\n3. Recalculate using manual court calibration tool")
    homo_mode = input("Select an option (1/2/3): ").strip()

    calib_dir = base_out / "calib_out"
    calib_dir.mkdir(parents=True, exist_ok=True)

    if homo_mode == "3":
        run_manual_calibration(str(video_path), str(calib_dir), calib_target_frame, logger)
        _, K, rvec, tvec = load_calibration(str(calib_dir))
        H_court = build_court_homography(K, rvec, tvec)
        np.save(calib_dir / "H_court.npy", H_court)
    elif homo_mode == "2":
        try: _, K, rvec, tvec = load_calibration(str(calib_dir))
        except Exception: 
            run_manual_calibration(str(video_path), str(calib_dir), calib_target_frame, logger)
            _, K, rvec, tvec = load_calibration(str(calib_dir))
        
        df = pd.read_csv("shuttleset/set/homography.csv")
        row = df[df['video'] == match_folder]
        H_court = np.array(ast.literal_eval(row.iloc[0]['homography_matrix'].replace(' ', ',')), dtype=np.float64) if not row.empty else build_court_homography(K, rvec, tvec)
    else:
        _, K, rvec, tvec = load_calibration(str(calib_dir))
        H_court = np.load(calib_dir / "H_court.npy") if (calib_dir / "H_court.npy").exists() else build_court_homography(K, rvec, tvec)

    print("\n═ SHUTTLE DETECTION ═")
    shuttle_cache = shuttle_dir / "shuttle_trimmed.npy"
    use_shuttle_cache = False
    if shuttle_cache.exists():
        if input(f"Cached shuttle detections found. Use it? (y/n): ").strip().lower() == 'y': use_shuttle_cache = True
            
    if use_shuttle_cache: trimmed_shuttle = np.load(shuttle_cache)
    else:
        csv_path = run_tracknet(video_path=trimmed_video_path, weights_dir=str(TRACKNET_DIR), raw_out_dir=str(shuttle_dir), eval_mode="weight", batch_size=4, large_video=True)
        trimmed_shuttle = parse_tracknet_csv(csv_path, trimmed_video_path) 
        np.save(shuttle_cache, trimmed_shuttle)

    if len(trimmed_shuttle) < n_trimmed: trimmed_shuttle = np.vstack([trimmed_shuttle, np.full((n_trimmed - len(trimmed_shuttle), 2), np.nan)])
    
    if input("Do you want to dump the annotated video for SHUTTLE DETECTION? (y/n): ").strip().lower() == 'y':
        dump_shuttle_video(trimmed_video_path, trimmed_shuttle[:n_trimmed], str(shuttle_dir / f"{match_folder}_shuttle_trimmed.mp4"))

    print("\n═ POSE ESTIMATION ═")
    pose_cache = pose_dir / "poses_trimmed.pkl"
    use_pose_cache = False
    if pose_cache.exists():
        if input(f"Cached pose estimations found. Use it? (y/n): ").strip().lower() == 'y': use_pose_cache = True

    if use_pose_cache:
        with open(pose_cache, "rb") as f: trimmed_poses = pickle.load(f)
    else:
        cap = cv2.VideoCapture(trimmed_video_path)
        det_m, pose_m, _ = load_pose_backend()
        trimmed_poses = []
        batch_frames = []
        batch_size = 32
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            batch_frames.append(frame)
            if len(batch_frames) == batch_size:
                batch_res = estimate_poses_batched(batch_frames, K, rvec, tvec, det_m, pose_m, batch_size=batch_size)
                trimmed_poses.extend(batch_res)
                batch_frames.clear()
                
        if len(batch_frames) > 0:
            batch_res = estimate_poses_batched(batch_frames, K, rvec, tvec, det_m, pose_m, batch_size=len(batch_frames))
            trimmed_poses.extend(batch_res)
            batch_frames.clear()
            
        cap.release()
        with open(pose_cache, "wb") as f: pickle.dump(trimmed_poses, f)
        
    if input("Do you want to dump the annotated video for POSE ESTIMATION? (y/n): ").strip().lower() == 'y':
        dump_pose_video(trimmed_video_path, trimmed_poses, str(pose_dir / f"{match_folder}_pose_trimmed.mp4"), H_court=H_court)

    global_poses = [None] * n_total
    for i, orig_f in enumerate(frame_map):
        if i < len(trimmed_poses): global_poses[orig_f] = trimmed_poses[i]

    print("\n═ SHOT CLASSIFICATION ═")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    weight_path = Path("weights/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt")
    if not weight_path.exists(): sys.exit(1)
        
    net = BST_CG_AP(in_dim=72, seq_len=100, n_class=25).to(device)
    net.load_state_dict(torch.load(str(weight_path), map_location=device, weights_only=True))
    net.eval()
    
    all_bst_types, bone_pairs = get_merged_stroke_types(), get_bone_pairs("coco")
    safe_trimmed_poses = [pf if pf is not None else EmptyPoseFrame() for pf in trimmed_poses]
    
    joints, pos_norm, shuttle_norm, _ = normalize_rally_data(safe_trimmed_poses, trimmed_shuttle, v_w, v_h, n_trimmed, H_court)

    results = []
    for i, row in gt_df.iterrows():
        hit_idx = int(row["frame_num"])
        gt_type = row["gt_type"]
        gt_mapped_stroke = SHUTTLESET_18_TO_BST_12.get(gt_type, "Unknown")
        
        if hit_idx not in frame_map: continue
        trimmed_hit_idx = frame_map.index(hit_idx)
        
        if row["skip"]:
            results.append({"rally": row.get("rally"), "set": row.get("set_file"), "frame_num": hit_idx, "gt_type": gt_type, "gt_mapped_stroke": gt_mapped_stroke, "gt_side": row.get("hitter_side_gt", "Unknown"), "bst_pred_full": "SKIP", "bst_pred_stroke": "SKIP", "bst_pred_side": "SKIP", "confidence": 0.0})
            continue

        curr_rally, curr_set = row["rally"], row["set_file"]
        rally_hits = gt_df[(gt_df["rally"] == curr_rally) & (gt_df["set_file"] == curr_set)]["frame_num"].tolist()
        hit_pos = rally_hits.index(hit_idx)

        prev_h = frame_map.index(rally_hits[hit_pos - 1]) if hit_pos > 0 and rally_hits[hit_pos - 1] in frame_map else max(0, trimmed_hit_idx - 50)
        next_h = frame_map.index(rally_hits[hit_pos + 1]) if hit_pos < len(rally_hits) - 1 and rally_hits[hit_pos + 1] in frame_map else min(n_trimmed - 1, trimmed_hit_idx + 50)

        j_win, p_win, s_win = np.zeros((100, 2, 17, 2), dtype=np.float32), np.zeros((100, 2, 2), dtype=np.float32), np.zeros((100, 2), dtype=np.float32)

        for offset in range(-50, 50):
            t_frame = trimmed_hit_idx + offset
            target_idx = offset + 50
            if prev_h <= t_frame <= next_h and 0 <= t_frame < n_trimmed:
                j_win[target_idx] = joints[t_frame]
                p_win[target_idx] = pos_norm[t_frame]
                s_win[target_idx] = shuttle_norm[t_frame]

        hp_in = np.concatenate((j_win, create_bones(j_win, bone_pairs)), axis=-2) 
        with torch.no_grad():
            logits = net(torch.tensor(hp_in).unsqueeze(0).to(device).view(1, 100, 2, -1), torch.tensor(s_win).unsqueeze(0).to(device), torch.tensor(p_win).unsqueeze(0).to(device), torch.tensor([100]).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        sorted_idx = np.argsort(probs)[::-1]
        expected_side = row.get("hitter_side_gt", "Unknown")
        pred_full, pred_side, pred_stroke, confidence = "Unknown", "Unknown", "Unknown", 0.0
        
        for idx in sorted_idx:
            cand_full = all_bst_types[idx]
            cand_conf = float(probs[idx])
            cand_side, cand_stroke = cand_full.split("_", 1) if "_" in cand_full else ("Unknown", "Unknown" if cand_full in ["Unknown", "未知球種"] else cand_full)
                
            if cand_stroke == "Unknown" and len(sorted_idx) > 1: continue
                
            if expected_side in ["Top", "Bottom"]:
                if cand_side == expected_side:
                    pred_full, pred_side, pred_stroke, confidence = cand_full, cand_side, cand_stroke, cand_conf
                    break
            else:
                pred_full, pred_side, pred_stroke, confidence = cand_full, cand_side, cand_stroke, cand_conf
                break

        results.append({"rally": row.get("rally"), "set": row.get("set_file"), "frame_num": hit_idx, "gt_type": gt_type, "gt_mapped_stroke": gt_mapped_stroke, "gt_side": expected_side, "bst_pred_full": pred_full, "bst_pred_side": pred_side, "bst_pred_stroke": pred_stroke, "confidence": confidence})

    out_df = pd.DataFrame(results)
    out_df.to_csv(shot_dir / "bst_shuttleset_results.csv", index=False, encoding='utf-8-sig')
    compute_metrics(out_df, shot_dir, logger)

    if input("\nDump Final BST Annotated Video? (y/n): ").strip().lower() == 'y':
        dump_final_annotated_video(trimmed_video_path, shot_dir / f"{match_folder}_final_annotated_trimmed.mp4", trimmed_shuttle, safe_trimmed_poses, out_df, frame_map, logger, H_court=H_court)

if __name__ == "__main__": main()