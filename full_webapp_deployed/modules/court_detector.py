import os
import logging
import numpy as np
from scipy.optimize import minimize
from inference_sdk import InferenceHTTPClient
import dotenv
import cv2
from config import COURT_POINTS_3D

dotenv.load_dotenv()  

logger = logging.getLogger(__name__)

class CourtDetector:
    def __init__(self):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY") 
        )
        self.workspace = os.getenv("ROBOFLOW_WORKSPACE")
        self.workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID")

    def detect(self, frame_path):
        """Calls Roboflow and extracts the most outer keypoints geometrically."""
        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": frame_path},
                use_cache=True 
            )
            return self._extract_outer_points(result[0])
        except Exception as e:
            logger.error(f"Court Detection Failed: {e}")
            return None
        
    def _extract_outer_points(self, data):
        output = {}
        
        # 1. Process Corners
        corner_preds = data.get("court_corner_predictions", {}).get("predictions", [])
        if len(corner_preds) >= 4:
            # First, identify which box is which based on box centers
            centers = np.array([[p['x'], p['y']] for p in corner_preds])
            
            # Helper to find the best point inside a specific corner box
            def get_extreme_point(box_idx, criteria):
                kps = corner_preds[box_idx].get('keypoints', [])
                if not kps: return centers[box_idx]
                pts = np.array([[kp['x'], kp['y']] for kp in kps])
                if criteria == "min_sum": return pts[np.argmin(pts.sum(axis=1))]
                if criteria == "max_sum": return pts[np.argmax(pts.sum(axis=1))]
                if criteria == "min_diff": return pts[np.argmin(pts[:, 0] - pts[:, 1])]
                if criteria == "max_diff": return pts[np.argmax(pts[:, 0] - pts[:, 1])]
                return centers[box_idx]

            # Assign roles to boxes based on centers
            # FarLeft (Top-Left): Min x+y
            idx_fl = np.argmin(centers.sum(axis=1))
            output["FarLeft"] = tuple(get_extreme_point(idx_fl, "min_sum").astype(int))
            
            # NearRight (Bottom-Right): Max x+y
            idx_nr = np.argmax(centers.sum(axis=1))
            output["NearRight"] = tuple(get_extreme_point(idx_nr, "max_sum").astype(int))
            
            # For FarRight and NearLeft, we look at the remaining two
            others = [i for i in range(len(corner_preds)) if i not in [idx_fl, idx_nr]]
            if len(others) >= 2:
                # FarRight (Top-Right): Max x-y
                idx_fr = others[0] if (centers[others[0], 0] - centers[others[0], 1]) > (centers[others[1], 0] - centers[others[1], 1]) else others[1]
                output["FarRight"] = tuple(get_extreme_point(idx_fr, "max_diff").astype(int))
                
                # NearLeft (Bottom-Left): Min x-y
                idx_nl = others[1] if idx_fr == others[0] else others[0]
                output["NearLeft"] = tuple(get_extreme_point(idx_nl, "min_diff").astype(int))

        # 2. Process Netlines (Topmost points)
        net_preds = data.get("predictions", {}).get("predictions", [])
        net_tops = []
        for net in net_preds:
            if net['class'] == "Netline":
                kps = net.get('keypoints', [])
                if kps:
                    top_kp = min(kps, key=lambda p: p['y'])
                    net_tops.append((int(top_kp['x']), int(top_kp['y'])))
        
        if len(net_tops) >= 2:
            net_tops.sort(key=lambda p: p[0])
            output["UpperNetLeft"], output["UpperNetRight"] = net_tops[0], net_tops[1]

        return output
    
    def calibrate(self, points_2d_dict, img_size):
        w, h = img_size
        
        # 1. Align detected 2D points with fixed 3D coordinates
        # Must match the order in court_calibration.py WORLD_PTS
        order = ["FarLeft", "FarRight", "NearLeft", "NearRight", "UpperNetLeft", "UpperNetRight"]
        pts_2d = np.array([points_2d_dict[k] for k in order], dtype=np.float64)
        pts_3d = COURT_POINTS_3D 

        # 2. Define Internal Optimizer for Intrinsics (optimise_K logic)
        def reprojection_error(p):
            fx, fy, cx, cy = p
            K_temp = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            ok, rv, tv = cv2.solvePnP(pts_3d, pts_2d, K_temp, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return 1e9
            proj, _ = cv2.projectPoints(pts_3d, rv, tv, K_temp, None)
            return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts_2d, axis=1)))

        # Initial guess and bounds for the optimizer
        x0 = [w * 2.0, w * 2.0, w / 2.0, h / 2.0]
        bounds = [
            (w * 0.8, w * 5.0),   # fx
            (w * 0.8, w * 5.0),   # fy
            (w * 0.35, w * 0.65), # cx
            (h * 0.35, h * 0.65), # cy
        ]

        res = minimize(reprojection_error, x0, method="L-BFGS-B", bounds=bounds,
                    options={"ftol": 1e-10, "maxiter": 3000})
        
        fx, fy, cx, cy = res.x
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print("solvePnP failed. Check point order.")
            return None

        rvec, tvec = cv2.solvePnPRefineLM(pts_3d, pts_2d, K, None, rvec, tvec)

        R, _ = cv2.Rodrigues(rvec)
        P = K @ np.hstack([R, tvec])
        P /= P[2, 3] # Normalize so P[2,3] = 1

        # Final error for logging
        final_error = res.fun
        print(f"Calibration Complete. Optimized Error: {final_error:.4f} px")

        return {
            "K": K.tolist(),
            "P": P.tolist(),
            "rvec": rvec.tolist(),
            "tvec": tvec.tolist(),
            "reproj_error": float(final_error)
        }