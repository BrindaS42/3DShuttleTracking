import os
import subprocess
import sys

# --- 1. ROBUST RUNTIME INSTALLATION ---
def install_dependencies():
    # 1. Essential build tools
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools==69.5.1", "wheel", "openmim"])

    # 2. Install MMCV 
    try:
        import mmcv
        print("MMCV already installed.")
    except ImportError:
        import torch
        torch_v = torch.__version__.split('+')[0]
        cuda_v = "cu121" if torch.version.cuda and "12" in torch.version.cuda else "cu118"
        url = f"https://download.openmmlab.com/mmcv/dist/{cuda_v}/torch{torch_v}/index.html"
        print(f"Installing MMCV for Torch {torch_v}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mmcv==2.1.0", "--no-build-isolation", "-f", url])

    # 3. Install MMengine, MMDet, and MMPose 
    try:
        import mmdet, mmpose
        print("MMLab core libraries already installed.")
    except ImportError:
        print("Installing MMLab core libraries...")
        subprocess.check_call([sys.executable, "-m", "mim", "install", "mmengine>=0.10.0"])
        subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet>=3.0.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mmpose>=1.1.0", "--no-deps"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xtcocotools", "json-tricks", "munkres", "chumpy-fork"])

# Execute installation 
install_dependencies()

# --- 2. IMPORT LIBRARIES ---
import json
import cv2
from mmengine.registry import init_default_scope
import pickle
import numpy as np
import gradio as gr
import torch
from dataclasses import dataclass, field
from typing import Optional
import torch
from scipy.optimize import minimize

from mmdet.utils import register_all_modules as register_mmdet
from mmpose.utils import register_all_modules as register_mmpose
# Re-import after installation to ensure registries are initialized
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown

# --- 2.  DATA STRUCTURES & CONFIG ---
COURT_W, COURT_L, NET_Y, POST_H = 6.7, 13.4, 6.7, 1.55
WORLD_PTS = np.array([
    [0.0, 0.0, 0.0], [COURT_W, 0.0, 0.0], [COURT_W, COURT_L, 0.0],
    [0.0, COURT_L, 0.0], [0.0, NET_Y, POST_H], [COURT_W, NET_Y, POST_H]
], dtype=np.float64)

_L_ANKLE, _R_ANKLE, _L_HIP, _R_HIP = 15, 16, 11, 12

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

# --- 3.  GEOMETRY ENGINE ---
def optimise_K(world_pts, image_pts, img_w, img_h):
    """Refines focal length and principal point independently (Non-square pixel fix)."""
    def err(p):
        fx, fy, cx, cy = p
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        ok, rv, tv = cv2.solvePnP(world_pts, image_pts, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok: return 1e9
        proj, _ = cv2.projectPoints(world_pts, rv, tv, K, None)
        return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1)))
    
    x0 = [img_w * 2.0, img_w * 2.0, img_w / 2.0, img_h / 2.0]
    bounds = [(img_w*0.8, img_w*5.0), (img_w*0.8, img_w*5.0), (img_w*0.35, img_w*0.65), (img_h*0.35, img_h*0.65)]
    res = minimize(err, x0, method="L-BFGS-B", bounds=bounds)
    fx, fy, cx, cy = res.x
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def backproject_to_floor(u, v, K, rvec, tvec, target_z=0.0):
    """Projects pixel to 3D world coordinates at a specific height."""
    R, _ = cv2.Rodrigues(rvec)
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    Rt = R.T
    tv = tvec.flatten()
    den = (Rt @ ray)[2]
    if abs(den) < 1e-10: return None
    lam = (target_z + (Rt @ tv)[2]) / den
    return (Rt @ (ray * lam - tv)).tolist()

# --- 4.  INFERENCE ENGINE ---
class PoseAPI:
    def __init__(self, yolo_path="best.pt"):
        from ultralytics import YOLO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Detection
        det_cfg = os.path.join(os.path.dirname(__import__('mmdet').__file__), '.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py')
        self.det_m = init_detector(det_cfg, "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth", device=self.device)
        
        # Initialize Pose
        pose_cfg = os.path.join(os.path.dirname(__import__('mmpose').__file__), '.mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py')
        self.pose_m = init_model(pose_cfg, "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth", device=self.device)
        
        self.yolo = YOLO(yolo_path)

    def run(self, video_path, img_pts, batch_size=24):
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        
        # 1.  Calibration Pipeline
        K = optimise_K(WORLD_PTS, img_pts, w, h)
        _, rv, tv = cv2.solvePnP(WORLD_PTS, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        rv, tv = cv2.solvePnPRefineLM(WORLD_PTS, img_pts, K, None, rv, tv)
        court_poly = img_pts[:4].astype(np.int32)

        results = []
        last_n, last_f = None, None
        f_idx = 0

        while True:
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            if not frames: break

            # --- SCOPE SWITCH: MMDET ---
            init_default_scope('mmdet')
            det_results = inference_detector(self.det_m, frames)
            
            # --- SCOPE SWITCH: MMPOSE ---
            init_default_scope('mmpose')

            for j, frame in enumerate(frames):
                pf = PoseFrame(f_idx)
                y_res = self.yolo.predict(frame, conf=0.3, verbose=False)[0].boxes.xyxy.cpu().numpy()
                inst = det_results[j].pred_instances
                r_boxes = inst.bboxes.cpu().numpy()[(inst.scores.cpu().numpy() > 0.45) & (inst.labels.cpu().numpy() == 0)]
                
                final_bboxes = []
                for yb in y_res:
                    ious = [self.get_iou(yb, rb) for rb in r_boxes]
                    if ious and max(ious) > 0.1: final_bboxes.append(r_boxes[np.argmax(ious)])

                if final_bboxes:
                    poses = inference_topdown(self.pose_m, frame, bboxes=np.array(final_bboxes))
                    cands = []
                    for k, res in enumerate(poses):
                        kp, cn = res.pred_instances.keypoints[0], res.pred_instances.keypoint_scores[0]
                        if self.is_valid(kp, cn, final_bboxes[k], court_poly, last_n) or self.is_valid(kp, cn, final_bboxes[k], court_poly, last_f):
                            idx = _L_ANKLE if cn[_L_ANKLE] > cn[_R_ANKLE] else _R_ANKLE
                            f3d = backproject_to_floor(kp[idx][0], kp[idx][1], K, rv, tv, 0.0)
                            cands.append({
                                'kp': kp.tolist(), 'box': final_bboxes[k].tolist(), 'cn': cn.tolist(),
                                'f3d': f3d, 'conf': float(np.mean(cn[[_L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE]]))
                            })

                    if cands:
                        c_n = max(cands, key=lambda x: self.get_iou(x['box'], last_n)) if last_n is not None else max(cands, key=lambda x: x['box'][3])
                        pf.near = PlayerPose(keypoints=c_n['kp'], bbox=c_n['box'], floor_pos_3d=c_n['f3d'], confidence=c_n['conf'])
                        last_n = c_n['box']
                        
                        cands = [c for c in cands if c['box'] != c_n['box']]
                        if cands:
                            c_f = max(cands, key=lambda x: self.get_iou(x['box'], last_f)) if last_f is not None else min(cands, key=lambda x: x['box'][3])
                            pf.far = PlayerPose(keypoints=c_f['kp'], bbox=c_f['box'], floor_pos_3d=c_f['f3d'], confidence=c_f['conf'])
                            last_f = c_f['box']

                results.append(pf)
                f_idx += 1
        cap.release()
        return results

    def get_iou(self, b1, b2):
        x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        if x2 < x1 or y2 < y1: return 0.0
        it = (x2-x1)*(y2-y1)
        return it / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - it + 1e-6)

    def is_valid(self, kp, cn, box, poly, last):
        if last is not None and self.get_iou(box, last) > 0.15: return True
        return any(cn[i] > 0.3 and cv2.pointPolygonTest(poly, (float(kp[i,0]), float(kp[i,1])), False) >= 0 for i in [_L_ANKLE, _R_ANKLE])

# --- 5. GRADIO INTERFACE ---
api_engine = None

def analyze(video, points_json):
    global api_engine
    if api_engine is None: api_engine = PoseAPI()
    
    try:
        img_pts = np.array(json.loads(points_json), dtype=np.float64)
        if len(img_pts) != 6: raise ValueError("Must provide exactly 6 points.")
    except Exception as e:
        return f"Error parsing JSON: {str(e)}"
        
    results = api_engine.run(video, img_pts)
    
    output_path = "poses.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    return output_path

demo = gr.Interface(
    fn=analyze, 
    inputs=[gr.Video(), gr.Textbox(label="6 Point JSON [[x,y],...]", placeholder="[[100,200], ...]")], 
    outputs=gr.File(label="Pose Data"),
    api_name="get_poses"
)

if __name__ == "__main__":
    demo.launch()