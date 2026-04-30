import os
import csv
import subprocess
import shutil
import uuid
import gradio as gr
from pathlib import Path

# Configuration
TRACKNET_V3_DIR = Path("tracknetv3")
PREDICT_SCRIPT = TRACKNET_V3_DIR / "predict_x.py"
WEIGHTS_TRACKNET = TRACKNET_V3_DIR / "TrackNet_best.pt"
WEIGHTS_INPAINT = TRACKNET_V3_DIR / "InpaintNet_best.pt"

def detect_shuttlecock(video_file):
    if video_file is None:
        return {"error": "No video provided"}

    # 1. Setup unique workspace for this request
    request_id = str(uuid.uuid4())[:8]
    temp_dir = Path(f"temp_{request_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded video to temp workspace
    video_path = temp_dir / "input_video.mp4"
    shutil.copy(video_file, video_path)
    
    # 2. Build the Command (Matching your shuttle_detection.py logic)
    # We use nonoverlap for speed if it's just an API, 
    # but 'weight' is more accurate.
    cmd = [
        "python", str(PREDICT_SCRIPT),
        "--video_file", str(video_path.resolve()),
        "--tracknet_file", str(WEIGHTS_TRACKNET.resolve()),
        "--inpaintnet_file", str(WEIGHTS_INPAINT.resolve()),
        "--save_dir", str(temp_dir.resolve()),
        "--eval_mode", "weight",
        "--batch_size", "16"
    ]

    try:
        # 3. Execute TrackNetV3
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # 4. Locate and Parse the Output CSV
        # TrackNetV3 auto-names it: {video_stem}_ball.csv
        csv_path = temp_dir / "input_video_ball.csv"
        
        if not csv_path.exists():
            return {"error": "Model failed to generate CSV output"}

        detections = []
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                detections.append({
                    "frame": int(row['Frame']),
                    "vis": int(row['Visibility']),
                    "x": int(float(row['X'])),
                    "y": int(float(row['Y']))
                })

        return {"detections": detections, "total_frames": len(detections)}

    except subprocess.CalledProcessError as e:
        return {"error": f"Subprocess error: {e.stderr}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # 5. Cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# Define Gradio Interface for API
iface = gr.Interface(
    fn=detect_shuttlecock,
    inputs=gr.Video(label="Upload Badminton Video"),
    outputs=gr.JSON(label="Frame Detections"),
    title="TrackNetV3 Shuttlecock Detection API",
    description="Upload a video to receive frame-by-frame (x, y) coordinates of the shuttlecock."
)

if __name__ == "__main__":
    iface.launch()