# """
# verify_3d_validity.py
# ═════════════════════
# Calculates a '3D Confidence Score' by checking:
#   1. Net Clearance (Geometric)
#   2. Pose Alignment (Positional Anchor)
#   3. Velocity Decay (Physics)
#   4. Court Boundary (Logical)
# """

# import numpy as np
# import pickle
# from pathlib import Path
# import sys

# # Load your project configs and geometry helpers
# sys.path.insert(0, str(Path(__file__).parent))
# from config import COURT_L, COURT_W, PLAYER_HEIGHT
# from court_calibration import load_calibration

# def verify_3d(traj_path, pose_path, calib_dir, hitter_side="near"):
#     print(f"\n{'═'*60}\n3-D COORDINATE VALIDITY REPORT\n{'═'*60}")
    
#     # 1. Load Data
#     traj = np.load(traj_path)  # N x 3 array
#     with open(pose_path, "rb") as f:
#         poses = pickle.load(f)
    
#     fps = 30.0 # Standard; adjust if different
#     N = len(traj)
#     dt = 1.0 / fps
    
#     # 2. NET CLEARANCE CHECK (The 'Physical Obstacle' Test)
#     # The net is at Y = COURT_L / 2
#     net_y = COURT_L / 2.0
#     # Find frame where shuttle crosses the net
#     y_vals = traj[:, 1]
#     net_idx = np.argmin(np.abs(y_vals - net_y))
#     z_at_net = traj[net_idx, 2]
    
#     net_status = "PASS" if z_at_net >= 1.52 else "FAIL" # 1.524m is standard net height
#     print(f"[1] Net Clearance: {z_at_net:.2f}m at the net line (Y={net_y:.2f}m)")
#     print(f"    Status: {net_status} (Expected Z > 1.52m)")

#     # 3. POSE ALIGNMENT (The 'Anchor' Test)
#     # Trajectory should start near the hitter and end near the receiver
#     hitter_pose = poses[0].near if hitter_side == "near" else poses[0].far
#     receiver_pose = poses[-1].far if hitter_side == "near" else poses[-1].near
    
#     # Extract 3D floor positions
#     h_pos = np.array(hitter_pose.floor_pos_3d) if hitter_pose.floor_pos_3d else None
#     r_pos = np.array(receiver_pose.floor_pos_3d) if receiver_pose.floor_pos_3d else None
    
#     if h_pos is not None:
#         # Distance between traj start (X,Y) and player (X,Y)
#         dist_h = np.linalg.norm(traj[0, :2] - h_pos[:2])
#         print(f"[2] Hitter Alignment: Dist = {dist_h:.2f}m")
    
#     if r_pos is not None:
#         dist_r = np.linalg.norm(traj[-1, :2] - r_pos[:2])
#         print(f"[3] Receiver Alignment: Dist = {dist_r:.2f}m")

#     # 4. VELOCITY DECAY (The 'Physics' Test)
#     # Shuttles MUST decelerate significantly due to high drag
#     vels = np.linalg.norm(np.diff(traj, axis=0), axis=1) * fps
#     v_start = vels[0]
#     v_end = vels[-1]
#     decay = (v_start - v_end) / v_start
    
#     phys_status = "PASS" if decay > 0.3 else "WARNING" # Expect at least 30% drop
#     print(f"[4] Velocity Profile: {v_start:.1f} m/s -> {v_end:.1f} m/s")
#     print(f"    Decay: {decay*100:.1f}% (Status: {phys_status})")

    

#     # 5. SUMMARY SCORE
#     score = 0
#     if net_status == "PASS": score += 40
#     if phys_status == "PASS": score += 30
#     if h_pos is not None and dist_h < 1.5: score += 15
#     if r_pos is not None and dist_r < 2.5: score += 15

#     # Y RANGE CHECK
#     y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
#     y_span = y_max - y_min
#     y_status = "PASS" if y_span > 7.0 else "FAIL"
#     print(f"[5] Y-axis span: {y_min:.2f}m → {y_max:.2f}m  (span={y_span:.2f}m)")
#     print(f"    Status: {y_status} (Expected span > 7.0m for full-court shot)")
#     if y_status == "PASS": score += 15  # add to scoring
    
#     print(f"{'─'*60}\nFINAL 3-D CONFIDENCE SCORE: {score}/115\n{'═'*60}")

# if __name__ == "__main__":
#     verify_3d(
#         "results/traj_out/trajectory_3d.npy",
#         "results/pose_out/poses.pkl",
#         "results/calib_out"
#     )
# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="hu2OVggSouSm6dPcPMqj"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="ritesh-3iojl",
    workflow_id="custom-workflow-4",
    images={
        "image": "./test_assets/test_image.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
# 4. Get your results
# save_path = "./test_assets/test_result.jpg"

# # Access the first element of the list, then save
# if result and len(result) > 0:
#     result[0].save(save_path)
#     print(f"Result saved to {save_path}")
# else:
#     print("No results returned from the workflow.")
import supervision as sv
import cv2
import base64
import numpy as np

if result and len(result) > 0:
    workflow_data = result[0]
    
    # 1. Grab the visualization data directly from the known key
    if "final_visualization" in workflow_data:
        viz_data = workflow_data["final_visualization"]
        
        # Roboflow sometimes returns a raw string, or a dict with a 'value' key
        base64_string = ""
        if isinstance(viz_data, dict) and "value" in viz_data:
            base64_string = viz_data["value"]
        elif isinstance(viz_data, str):
            base64_string = viz_data
            
        if base64_string:
            # 2. Decode the Base64 string into bytes
            img_data = base64.b64decode(base64_string)
            
            # 3. Convert bytes into a NumPy array
            img_array = np.frombuffer(img_data, np.uint8)
            
            # 4. Decode the array into an OpenCV image
            annotated_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 5. Save the final image
            save_path = "./test_assets/test_result_annotated.jpg"
            cv2.imwrite(save_path, annotated_img)
            print(f"Success! Annotated image extracted and saved to: {save_path}")
        else:
            print("Found 'final_visualization', but couldn't read the base64 string.")
    else:
        print("Could not find 'final_visualization' in the JSON keys.")
        print(f"Available keys: {workflow_data.keys()}")
else:
    print("No results returned.")