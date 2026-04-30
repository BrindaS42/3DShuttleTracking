import gradio as gr
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from new_pipeline import detect_swing_actions_yolo, trajectory_smoothing, compute_hd_t, assign_sides_pkl

def run_sra_inference(video_path, shuttle_path, pose_path):
    try:
        # 1. Load the Inputs
        # Load Poses
        with open(pose_path, 'rb') as f:
            poses = pickle.load(f)
        
        # Load Shuttle Trajectory
        shuttle_2d = np.load(shuttle_path)
        
        # Setup metadata
        frame_ids = list(range(len(poses)))
        sides = assign_sides_pkl(poses)
        
        # 2. Path A: Trajectory Analysis (HD-T)
        shuttle_smooth = trajectory_smoothing(shuttle_2d)
        hdt_f, _ = compute_hd_t(shuttle_smooth, frame_ids, min_dist=12)
        
        # 3. Path B: Visual Swing Detection (HD-A)
        # Note: detect_swing_actions_yolo usually needs the video path 
        # to run the classifier on the frames
        raw_f, raw_p, raw_c = detect_swing_actions_yolo(poses, frame_ids, sides, video_path)
        
        # 4. SRA Fusion Logic (The "Precision-Locked" Strategy)
        final_hits = []
        used_hdt = set()
        
        # Sort by frame to process chronologically
        data = sorted(zip(raw_f, raw_p, raw_c), key=lambda x: x[0])
        
        for f_y, p_y, c_y in data:
            # Physical Window: Does the shuttle reverse near this swing?
            window = [f_y - 12, f_y + 4]
            possible_hdt = [f for f in hdt_f if window[0] <= f <= window[1] and f not in used_hdt]
            
            if possible_hdt:
                # Path A: Trajectory confirmed (Highest Confidence)
                best_f = min(possible_hdt, key=lambda f: abs(f - (f_y - 3)))
                final_hits.append({
                    "frame": int(best_f), 
                    "player": int(p_y), 
                    "confidence": float(c_y), 
                    "method": "Trajectory-Confirmed"
                })
                used_hdt.add(best_f)
            elif c_y > 0.95:
                # Path B: Recovery (YOLO is extremely sure)
                final_hits.append({
                    "frame": int(f_y - 2), 
                    "player": int(p_y), 
                    "confidence": float(c_y), 
                    "method": "Visual-Recovery"
                })

        # 5. Temporal De-duplication (0.25s / 15 frame buffer)
        final_hits = sorted(final_hits, key=lambda x: x['frame'])
        deduped = []
        for h in final_hits:
            if not deduped or (h['frame'] - deduped[-1]['frame']) > 15:
                deduped.append(h)
        
        # 6. Export to CSV
        output_df = pd.DataFrame(deduped)
        output_path = "segmented_shots.csv"
        output_df.to_csv(output_path, index=False)
        
        return output_path

    except Exception as e:
        return f"Error during processing: {str(e)}"

# --- Gradio UI Setup ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# 🏸 Badminton SRA Shot Segmenter")
    gr.Markdown("Upload your pre-processed files to fuse visual swing data with shuttle trajectories.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Trimmed Video (.mp4)")
            shuttle_input = gr.File(label="Shuttle Trajectory (.npy)")
            pose_input = gr.File(label="Poses (.pkl)")
            submit_btn = gr.Button("Generate Shot Segmentation", variant="primary")
        
        with gr.Column():
            csv_output = gr.File(label="Download Segmented Shots CSV")

    submit_btn.click(
        fn=run_sra_inference,
        inputs=[video_input, shuttle_input, pose_input],
        outputs=[csv_output]
    )

if __name__ == "__main__":
    demo.launch()