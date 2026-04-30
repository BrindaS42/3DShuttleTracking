import cv2
import logging
import os
import subprocess
import numpy as np
import time
import ffmpeg  

logger = logging.getLogger(__name__)

def draw_court_keypoints(frame, points_dict):

    if not points_dict:
        return frame
        
    annotated = frame.copy()
    
    # Define colors (BGR) and point types
    colors = {
        "Corner": (0, 255, 0),    # Green for court corners
        "Net": (0, 255, 255)      # Yellow for net tops
    }
    
    # Mapping points to their display groups
    groups = {
        "FarLeft": "Corner", "FarRight": "Corner", 
        "NearLeft": "Corner", "NearRight": "Corner",
        "UpperNetLeft": "Net", "UpperNetRight": "Net"
    }

    for name, coord in points_dict.items():
        if coord is None:
            continue
            
        x, y = coord
        group = groups.get(name, "Corner")
        color = colors.get(group)

        cv2.circle(annotated, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 10, (0, 0, 0), 2, cv2.LINE_AA)
        
        label = f"{name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(label, font, scale, thickness)
        
        cv2.rectangle(annotated, (x + 12, y - h - 5), (x + 12 + w, y + 5), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x + 12, y), font, scale, color, thickness, cv2.LINE_AA)

    logger.info("Frame annotation complete.")
    return annotated

def draw_shuttle_video(input_video_path, detections, output_dir=".", trail_len=20):
    timestamp = int(time.time())
    raw_temp = f"raw_temp.mp4"
    # Using a simple filename for the final output
    final_name = f"shuttle_out.mp4"
    final_path = os.path.join(output_dir, final_name)

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Standard OpenCV writer (not web-friendly)
    writer = cv2.VideoWriter(raw_temp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    det_map = {d['frame']: d for d in detections if d['vis'] == 1}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Draw Trail
        for i in range(max(0, frame_idx - trail_len), frame_idx):
            if i in det_map:
                alpha = (i - (frame_idx - trail_len)) / trail_len
                color = (0, int(150 * alpha), int(255 * alpha))
                cv2.circle(frame, (det_map[i]['x'], det_map[i]['y']), 3, color, -1, cv2.LINE_AA)
        
        if frame_idx in det_map:
            d = det_map[frame_idx]
            cv2.circle(frame, (d['x'], d['y']), 6, (0, 255, 0), 2, cv2.LINE_AA)
            
        writer.write(frame)
        frame_idx += 1
        
    cap.release()
    writer.release()

    try:
        (
            ffmpeg
            .input(raw_temp)
            .output(
                final_path, 
                vcodec='libx264', 
                pix_fmt='yuv420p', 
                movflags='+faststart', 
                crf=23,
                preset='fast'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        if os.path.exists(raw_temp): 
            os.remove(raw_temp)
    except ffmpeg.Error as e:
        print("FFmpeg Python Error:", e.stderr.decode())
        # Fallback if it fails
        os.rename(raw_temp, final_path)
    
    return os.path.abspath(final_path)

def draw_pose_video(input_video_path, pose_data, output_dir="."):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    raw_temp = "raw_pose_temp.mp4"
    final_path = os.path.join(output_dir, "pose_annotated.mp4")
    writer = cv2.VideoWriter(raw_temp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # COCO Skeleton pairs
    skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), 
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), 
                (0, 2), (1, 3), (2, 4)]

    for f_idx, frame_data in enumerate(pose_data):
        ret, frame = cap.read()
        if not ret: break

        for player_type, color in [("near", (0, 255, 0)), ("far", (255, 0, 0))]:
            pose = getattr(frame_data, player_type)
            if not pose or pose.keypoints is None: continue

            # 1. Draw Bounding Box
            b = pose.bbox
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
            cv2.putText(frame, player_type.upper(), (int(b[0]), int(b[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 2. Draw Skeleton
            kp = np.array(pose.keypoints)
            for p1, p2 in skeleton:
                pt1 = tuple(kp[p1].astype(int))
                pt2 = tuple(kp[p2].astype(int))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

            # 3. Draw Joints
            for i, (x, y) in enumerate(kp):
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        writer.write(frame)

    cap.release()
    writer.release()
    
    # Re-encode for browser (using the same ffmpeg logic from your draw_shuttle_video)
    try:
        ffmpeg.input(raw_temp).output(final_path, vcodec='libx264', pix_fmt='yuv420p', 
                                     movflags='+faststart').overwrite_output().run(quiet=True)
        os.remove(raw_temp)
    except:
        os.rename(raw_temp, final_path)
    return final_path

def draw_trajectory_video(input_video_path, traj_2d_list, output_path):
    """Renders the final physics-estimated trajectory trail[cite: 24]."""
    cap = cv2.VideoCapture(input_video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    raw_temp = "temp_traj_render.mp4"
    writer = cv2.VideoWriter(raw_temp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    trail_len = 30
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Draw Trajectory Trail (Amber color as per source 24)
        for j in range(max(0, frame_idx - trail_len), frame_idx):
            if j < len(traj_2d_list) and traj_2d_list[j] is not None:
                alpha = (j - (frame_idx - trail_len)) / trail_len
                # Amber-Orange color: (0, 180, 255) in BGR
                color = (0, int(180 * alpha), int(255 * alpha))
                pos = tuple(map(int, traj_2d_list[j]))
                cv2.circle(frame, pos, 3, color, -1, cv2.LINE_AA)
        
        writer.write(frame)
        frame_idx += 1
        
    cap.release()
    writer.release()
    ffmpeg.input(raw_temp).output(output_path, vcodec='libx264', pix_fmt='yuv420p', movflags='+faststart').run(overwrite_output=True, quiet=True)
    if os.path.exists(raw_temp): os.remove(raw_temp)
    return output_path