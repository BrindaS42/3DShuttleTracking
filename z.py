"""
render_existing.py
══════════════════
Instantly renders the final annotated video using already cached 
shuttle_trimmed.npy, poses_trimmed.pkl, bst_shuttleset_results.csv, and trimmed_temp.mp4.
"""

import cv2
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path

BST_TO_ENGLISH = {
    "發短球": "Short Service", "發長球": "Long Service", "長球": "Clear",
    "殺球": "Smash", "切球": "Drop", "挑球": "Lob",
    "平球": "Drive", "放小球": "Net shot", "擋小球": "Return net", 
    "勾球": "Cross-court net", "推球": "Push", "撲球": "Rush", "Unknown": "Unknown"
}

class EmptyPoseFrame:
    near = None
    far = None

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
    return combined_df.sort_values("frame_num").reset_index(drop=True)

def rebuild_frame_map(match_folder, scope, n_total, buffer=60):
    gt_df = load_shuttleset_gt(match_folder, scope)
    required_frames = set()
    for _, group in gt_df.groupby(["set_file", "rally"]):
        first_hit = int(group["frame_num"].min())
        last_hit = int(group["frame_num"].max())
        for f in range(max(0, first_hit - buffer), min(n_total, last_hit + buffer + 1)):
            required_frames.add(f)
    return sorted(list(required_frames))

def draw_pose_frame(frame, pf_obj, coco_pairs):
    if not pf_obj: return
    players = []
    if hasattr(pf_obj, 'near') and pf_obj.near is not None: players.append(("Near", pf_obj.near))
    if hasattr(pf_obj, 'far') and pf_obj.far is not None: players.append(("Far", pf_obj.far))
    if isinstance(pf_obj, list):
        for i, p in enumerate(pf_obj): players.append((f"Player {i+1}", p))

    for label, person in players:
        kpts = None
        if hasattr(person, 'keypoints') and person.keypoints is not None: kpts = np.array(person.keypoints)
        elif isinstance(person, dict) and 'keypoints' in person: kpts = np.array(person['keypoints'])
        elif isinstance(person, (list, np.ndarray)): kpts = np.array(person)

        if kpts is not None and kpts.ndim >= 2 and kpts.shape[0] >= 17:
            for p1, p2 in coco_pairs:
                x1, y1, x2, y2 = int(kpts[p1][0]), int(kpts[p1][1]), int(kpts[p2][0]), int(kpts[p2][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x, y in kpts[:, :2]:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

def main():
    print("🎥 Instant Video Renderer from Cached Data 🎥")
    match_folder = input("Enter match folder name: ").strip()
    
    base_out = Path("test_shuttlenet") / match_folder
    shot_dir = base_out / "shot_out"
    
    if not shot_dir.exists():
        print(f"Directory {shot_dir} doesn't exist. Did you run check.py first?")
        return
        
    print("\n═ EXACT SCOPE YOU USED PREVIOUSLY ═")
    avail_sets = sorted([f for f in os.listdir(Path("shuttleset/set") / match_folder) if f.startswith("set")])
    print("1. Specific Set AND exact Rally Numbers")
    print("2. Specific Set (entire set)")
    print("3. ALL sets")
    c = input("Select option (1/2/3): ").strip()
    
    scope = {"sets": avail_sets, "rallies": None}
    if c in ['1', '2']:
        set_choice = input("Enter Set filename: ").strip()
        scope["sets"] = [set_choice] if set_choice in avail_sets else avail_sets
        if c == '1': scope["rallies"] = [int(r.strip()) for r in input("Enter Rally numbers (comma separated): ").split(',')]

    # Load resources
    print("Loading cached data...")
    orig_video_path = Path(f"test_assets/{match_folder}.mp4")
    cap_orig = cv2.VideoCapture(str(orig_video_path))
    n_total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_orig.release()
    
    frame_map = rebuild_frame_map(match_folder, scope, n_total)
    trimmed_shuttle = np.load(base_out / "shuttle_out" / "shuttle_trimmed.npy")
    with open(base_out / "pose_out" / "poses_trimmed.pkl", "rb") as f:
        trimmed_poses = pickle.load(f)
        
    results_df = pd.read_csv(shot_dir / "bst_shuttleset_results.csv")
    trimmed_video_path = str(base_out / "trimmed_temp.mp4")
    out_path = shot_dir / f"{match_folder}_final_annotated_trimmed.mp4"

    # Render
    print("Rendering video...")
    cap = cv2.VideoCapture(trimmed_video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    pred_dict = {
        row["frame_num"]: {
            "p_side": row["bst_pred_side"], "p_stroke": row["bst_pred_stroke"],
            "gt_side": row["gt_side"], "gt_stroke": row["gt_mapped_stroke"]
        } 
        for _, row in results_df.iterrows() if row["bst_pred_stroke"] != "SKIP"
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
            draw_pose_frame(frame, trimmed_poses[frame_idx], coco_pairs)
                            
        if orig_f in pred_dict:
            d = pred_dict[orig_f]
            p_en = BST_TO_ENGLISH.get(d["p_stroke"], d["p_stroke"])
            gt_en = BST_TO_ENGLISH.get(d["gt_stroke"], d["gt_stroke"])
            text_pred = f"Pred: {d['p_side']} | {p_en}"
            text_gt   = f"GT  : {d['gt_side']} | {gt_en}"
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
    print(f"✅ Video saved to {out_path}")

if __name__ == "__main__":
    main()