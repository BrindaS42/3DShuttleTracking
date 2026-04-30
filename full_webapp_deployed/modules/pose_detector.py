# modules/pose_detector.py
import os
import json
import pickle
import logging
import io
from dataclasses import dataclass, field
from typing import Optional
from gradio_client import Client, handle_file
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- 1. Class definitions MUST stay here ---
@dataclass
class PlayerPose:
    left_ankle_px:  Optional[list] = None
    right_ankle_px: Optional[list] = None
    keypoints:      Optional[list] = None   
    bbox:           Optional[list] = None   
    floor_pos_3d:   Optional[list] = None   
    body_pos_3d:    Optional[list] = None   
    confidence:     float          = 0.0

@dataclass
class PoseFrame:
    frame_idx: int
    near: PlayerPose = field(default_factory=PlayerPose)
    far:  PlayerPose = field(default_factory=PlayerPose)

# --- 2. THE CRITICAL FIX: Custom Unpickler ---
class PoseUnpickler(pickle.Unpickler):
    """Redirects __main__ lookups to this module's classes."""
    def find_class(self, module, name):
        if module == "__main__":
            if name == "PlayerPose": return PlayerPose
            if name == "PoseFrame": return PoseFrame
        return super().find_class(module, name)

class PoseDetector:
    def __init__(self):
        self.space_id = "TanmaySarda/Pose_and_bbox"
        self.token = os.getenv("HF_TOKEN")
        self.client = Client(self.space_id, token=self.token)

    def detect(self, video_path, points_dict):
        logger.info(f"Requesting Poses from {self.space_id}")
        order = ["FarLeft", "FarRight", "NearLeft", "NearRight", "UpperNetLeft", "UpperNetRight"]
        
        try:
            points_list = [[int(points_dict[k][0]), int(points_dict[k][1])] for k in order]
            points_json = json.dumps(points_list)
        except Exception as e:
            logger.error(f"Failed to format points: {e}")
            return None

        try:
            # --- 3. DEBUGGING: Check the API Response ---
            result_path = self.client.predict(
                video=handle_file(video_path),
                points_json=points_json,
                api_name="/get_poses"
            )
            
            print(f"\n--- POSE API DEBUG ---")
            print(f"DEBUG: result_path type: {type(result_path)}")
            print(f"DEBUG: result_path value: {result_path}")

            if result_path and os.path.exists(result_path):
                print(f"DEBUG: File exists! Size: {os.path.getsize(result_path)} bytes")
                with open(result_path, "rb") as f:
                    # Use the custom unpickler instead of standard pickle.load()
                    return PoseUnpickler(f).load()
            else:
                print(f"DEBUG: result_path is invalid or file missing.")
                return None
                
        except Exception as e:
            print(f"DEBUG: Pose API Exception occurred: {str(e)}")
            logger.error(f"Pose API Error: {e}")
            return None