import gradio as gr
import cv2
import os
import concurrent.futures
import numpy as np
import pickle
from modules.court_detector import CourtDetector
from modules.shuttle_detector import ShuttleDetector
from modules.pose_detector import PoseDetector
from modules.shot_segmentor import ShotSegmentor
from modules.trajectory_estimator import TrajectoryEstimator
from modules.shot_classifier import ShotClassifier
from utils.drawing_utils import draw_court_keypoints, draw_shuttle_video, draw_pose_video, draw_trajectory_video

court_mod = CourtDetector()
shuttle_mod = ShuttleDetector()
pose_mod = PoseDetector()
segmentor = ShotSegmentor()
traj_mod = TrajectoryEstimator()
classifier = ShotClassifier()

traj_3d_cache = gr.State({})

CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def save_to_disk(filename, data):
    """Saves data to a local pickle file for persistence."""
    with open(os.path.join(CACHE_DIR, filename), "wb") as f:
        pickle.dump(data, f)

def load_from_disk(filename):
    """Loads data from a local pickle file if it exists."""
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def update_frame(video_path, frame_idx):
    """Extracts a specific frame for the manual navigator."""
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def on_video_setup(video_path):
    if not video_path: return gr.update(maximum=100, value=0), None
    save_to_disk("v_path.pkl", video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return gr.update(maximum=total_frames - 1, value=0), update_frame(video_path, 0)

def run_classification_task(video_path, hit_frames, shuttle_cache, pose_cache, calib_cache, traj_3d_data):
    if not traj_3d_data:
        return [["Error", "Run Trajectory (Tab 3) first", "N/A"]]
    
    intervals = segmentor.get_intervals(hit_frames)
    
    # Prepare 2D shuttle array
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    s2d_all = np.full((total_frames, 2), np.nan)
    for det in shuttle_cache["detections"]:
        if det['vis'] == 1: s2d_all[det['frame']] = [det['x'], det['y']]
        
    return classifier.classify_rally_shots(video_path, intervals, s2d_all, pose_cache, calib_cache, traj_3d_data)

def run_heavy_pipeline(video_path):
    if not video_path: return
    
    # Use a dictionary to map every output component to its value
    # This satisfies the 9-output requirement of the click event
    out = {
        court_display: None, summary_table: None, pose_display: None,
        shuttle_display: None, shuttle_log: "⏳ Starting...",
        pts_cache: None, shuttle_cache: None, pose_cache: None, calib_cache: None
    }
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_court = executor.submit(process_court_only, video_path)
        f_shuttle = executor.submit(process_shuttle_task, video_path)

        # 1. Court results
        c_img, pts, table_data, calib_raw = f_court.result()
        out.update({
            court_display: c_img,
            summary_table: table_data,
            pts_cache: pts,
            calib_cache: calib_raw
        })
        yield out 

        # 2. Start Pose task after court[cite: 21]
        f_pose = executor.submit(process_pose_only, video_path, pts)
        
        pending = {f_shuttle, f_pose}
        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                if f == f_shuttle:
                    path, msg, shuttle_data_raw = f.result()
                    out.update({
                        shuttle_display: path,
                        shuttle_log: msg,
                        shuttle_cache: shuttle_data_raw
                    })
                elif f == f_pose:
                    path, pose_data_raw = f.result()
                    out.update({
                        pose_display: path,
                        pose_cache: pose_data_raw
                    })
            yield out # Yield the full 9-key dictionary again[cite: 21]

def process_court_only(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read(); cap.release()
    h, w = frame.shape[:2]
    temp = "temp_c.jpg"; cv2.imwrite(temp, frame)
    pts = court_mod.detect(temp)
    # Sanitize pts: Convert np.int64 to standard int to avoid JSON errors later
    pts = {k: (int(v[0]), int(v[1])) for k, v in pts.items()}
    calib = court_mod.calibrate(pts, (w, h))
    os.remove(temp)

    save_to_disk("pts.pkl", pts)
    save_to_disk("calib.pkl", calib)

    annotated = draw_court_keypoints(frame, pts)
    table = [[k, str(v), f"{calib['reproj_error']:.4f} px"] for k, v in pts.items()]
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), pts, table, calib

def process_pose_only(video_path, pts):
    data = pose_mod.detect(video_path, pts)
    save_to_disk("pose.pkl", data)
    path = draw_pose_video(video_path, data)
    return path, data

def process_shuttle_task(video_path):
    data = shuttle_mod.detect(video_path)
    if data:
        save_to_disk("shuttle.pkl", data)
    path = draw_shuttle_video(video_path, data["detections"]) if data else None
    return path, "✅ Shuttle Complete", data

def run_trajectory_pipeline(video_path, hit_frames, shuttle_cache, pose_cache, calib_cache):
    if not shuttle_cache or not calib_cache:
        return None, [["Error", "Heavy processing data missing (Run Tab 1 first)", "0"]], {}
    
    intervals = segmentor.get_intervals(hit_frames)
    if not intervals:
        return None, [["Error", "No shots segmented in Tab 2", "0"]], {}

    all_traj_2d = []
    table_data = []
    
    # Reconstruct 2D detections from cache for the optimizer
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Get actual video FPS
    if fps <= 0: fps = 30.0
    cap.release()
    
    shuttle_2d_all = np.full((total_frames, 2), np.nan)
    for det in shuttle_cache["detections"]:
        if det['vis'] == 1: 
            shuttle_2d_all[det['frame']] = [det['x'], det['y']]

    P = np.array(calib_cache['P'])
    traj_3d_dict = {}
    for i, (start, end) in enumerate(intervals):
        # Use pose cache for world-space anchors
        h_3d = pose_cache[start].near.floor_pos_3d if pose_cache and start < len(pose_cache) else None
        r_3d = pose_cache[end].far.floor_pos_3d if pose_cache and end < len(pose_cache) else None
        
        # FIX: Added 'i+1' as the shot_id argument
        traj_3d, err = traj_mod.estimate_shot(
            i + 1,                       # shot_id
            shuttle_2d_all[start:end],   # shuttle_2d
            P,                           # P
            h_3d,                        # hitter_3d
            r_3d,                        # receiver_3d
            fps                          # fps
        )
        traj_3d_dict[i+1] = traj_3d
        # Project 3D trajectory to 2D for the final trail
        pts_homo = np.hstack([traj_3d, np.ones((len(traj_3d), 1))])
        proj = (P @ pts_homo.T).T
        all_traj_2d.extend(proj[:, :2] / proj[:, 2:3])
        table_data.append([f"Shot {i+1}", f"{start}-{end}", f"{err:.2f} px"])
        
    final_path = draw_trajectory_video(video_path, all_traj_2d, "traj_final.mp4")
    return final_path, table_data, traj_3d_dict

# --- UI Setup ---
with gr.Blocks() as demo:
    v_path_cached = load_from_disk("v_path.pkl")
    pts_cached = load_from_disk("pts.pkl")
    shuttle_cached = load_from_disk("shuttle.pkl")
    pose_cached = load_from_disk("pose.pkl")
    calib_cached = load_from_disk("calib.pkl")
    hits_cached = load_from_disk("hits.pkl") or []

    # 2. Initialize States[cite: 11]
    pts_cache = gr.State(pts_cached); shuttle_cache = gr.State(shuttle_cached)
    pose_cache = gr.State(pose_cached); calib_cache = gr.State(calib_cached)
    hit_frames = gr.State(hits_cached)
    traj_3d_storage = gr.State({})

    gr.Markdown("# 🏸 Supreme Badminton Analytics")
    # Restore video input if path exists[cite: 11]
    video_input = gr.Video(label="Input Rally Video", value=v_path_cached)
    
    with gr.Tabs():
        with gr.TabItem("1. Detection & Pose"):
            run_btn = gr.Button("🚀 Run Heavy Analysis", variant="primary")
            with gr.Row():
                with gr.Column():
                    court_display = gr.Image(label="Court")
                    summary_table = gr.Dataframe(headers=["Point", "Coords", "Error"])
                with gr.Column():
                    pose_display = gr.Video(label="Poses")
                    shuttle_display = gr.Video(label="Shuttle")
                    shuttle_log = gr.Textbox(label="Status")

        with gr.TabItem("2. Shot Segmentation"):
            gr.Markdown("### 🔍 Frame-by-Frame Navigator")
            with gr.Row():
                mark_btn = gr.Button("🎯 Mark Current Frame as Hit", variant="primary")
                clear_btn = gr.Button("🗑️ Clear List")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # PATCH: Initialize seg_display with the first frame of the cached video
                    init_frame = update_frame(v_path_cached, 0) if v_path_cached else None
                    seg_display = gr.Image(value=init_frame, label="Manual Navigator", interactive=False)
                    seg_slider = gr.Slider(0, 100, step=1, label="Seek Frame")
                with gr.Column(scale=1):
                    seg_summary = gr.Textbox(
                        label="Marked Hit List", 
                        value=f"Marked Frames: {hits_cached}" if hits_cached else "", 
                        interactive=False
                    )

        with gr.TabItem("3. Trajectory"):
            calc_traj_btn = gr.Button("🎯 Estimate Trajectory (Using Cache)", variant="primary")
            traj_display = gr.Video(label="Estimated Trajectory Video")
            traj_report = gr.Dataframe(headers=["Shot", "Frames", "Error"])

        with gr.TabItem("4. Shot Classification"):
            classify_btn = gr.Button("🧠 Analyze Shot Types (5D LF)", variant="primary")
            classification_table = gr.Dataframe(
                headers=["Shot Number", "Hitter Side", "Classification"],
                label="Shot Classification Report"
            )
    # 1. Update the slider range as soon as a video is uploaded[cite: 12]
    video_input.change(
        fn=on_video_setup, 
        inputs=[video_input], 
        outputs=[seg_slider, seg_display]
    )

    # 2. Update the image preview as the slider moves[cite: 12]
    seg_slider.change(
        fn=update_frame, 
        inputs=[video_input, seg_slider], 
        outputs=[seg_display],
        show_progress="hidden" # Removes the flickering loading spinner[cite: 12]
    )

    # 3. Mark and Clear logic[cite: 14]
    def mark_and_save(f, current):
        new_list = sorted(list(set(current + [f])))
        save_to_disk("hits.pkl", new_list) # Persist the hits[cite: 11]
        return new_list, f"Marked Frames: {new_list}"

    mark_btn.click(fn=mark_and_save, inputs=[seg_slider, hit_frames], outputs=[hit_frames, seg_summary])
        
    clear_btn.click(lambda: ([], "Marked Frames: []"), None, [hit_frames, seg_summary])

    # Final logic mapping
    run_btn.click(
        fn=run_heavy_pipeline, 
        inputs=[video_input], 
        outputs=[court_display, summary_table, pose_display, shuttle_display, shuttle_log, 
                 pts_cache, shuttle_cache, pose_cache, calib_cache]
    )

    mark_btn.click(
        lambda f, current: (sorted(list(set(current + [f]))), str(sorted(list(set(current + [f]))))), 
        inputs=[seg_slider, hit_frames], 
        outputs=[hit_frames, seg_summary]
    )

    calc_traj_btn.click(
        fn=run_trajectory_pipeline, 
        inputs=[video_input, hit_frames, shuttle_cache, pose_cache, calib_cache],
        outputs=[traj_display, traj_report, traj_3d_storage]
    )

    classify_btn.click(
        fn=run_classification_task,
        inputs=[video_input, hit_frames, shuttle_cache, pose_cache, calib_cache, traj_3d_storage],
        outputs=[classification_table]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())