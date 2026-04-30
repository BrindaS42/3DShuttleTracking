import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from stroke.bst_5d import BST_CG_AP
from stroke.shuttleset import get_stroke_types

DEVICE = torch.device('cpu')
COURT_W, COURT_L, COURT_H = 6.1, 13.4, 5.0
COCO_BONES = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
              (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
ALL_TYPES = get_stroke_types()

# English Mapping for all 17 types[cite: 23, 25]
BST_TO_ENGLISH = {
    "放小球": "Net Shot", "擋小球": "Return Net", "殺球": "Smash", 
    "點扣": "Wrist Smash", "挑球": "Lob", "防守回挑": "Defensive Return Lob",
    "長球": "Clear", "平球": "Drive", "後場抽平球": "Back-court Drive", 
    "切球": "Drop", "過渡切球": "Passive Drop", "推球": "Push",
    "撲球": "Rush (Kill)", "防守回抽": "Defensive Return Drive", 
    "勾球": "Cross-court Net Shot", "發短球": "Short Service", 
    "發長球": "Long Service", "未知球種": "Unknown"
}

class ShotClassifier:
    def __init__(self, weight_path="stroke/best_bst_5d_global.pt"):
        self.model = BST_CG_AP(in_dim=72, seq_len=100, n_class=35).to(DEVICE)
        if Path(weight_path).exists():
            checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def classify_rally_shots(self, video_path, intervals, shuttle_2d_all, pose_cache, calib_cache, traj_3d_dict):
        results = []
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()

        for i, (start, end) in enumerate(intervals):
            shot_id = i + 1
            # 1. Determine Hitter Side from Pose/Trajectory data
            # If the shuttle at hit frame is closer to 'far' player (Top) or 'near' (Bottom)
            hitter_side = self._determine_hitter_side(start, pose_cache, shuttle_2d_all)
            
            # 2. Extraction & Prediction
            jnb, pos, s5d = self._extract_features(start, end, shuttle_2d_all, pose_cache, calib_cache, traj_3d_dict.get(shot_id), w, h)
            
            with torch.no_grad():
                logits = self.model(torch.tensor(jnb).unsqueeze(0), torch.tensor(s5d).unsqueeze(0), 
                                    torch.tensor(pos).unsqueeze(0), torch.tensor([100]))
                probs = F.softmax(logits, dim=1)[0].numpy()

            # 3. Post-Process: Select best match for the determined side
            sorted_indices = np.argsort(probs)[::-1]
            final_stroke_en = "Unknown"
            
            for idx in sorted_indices:
                cand_label = ALL_TYPES[idx]
                if cand_label == "未知球種": continue
                
                cand_side, cand_stroke_cn = cand_label.split("_", 1) if "_" in cand_label else ("Both", cand_label)
                
                # Check if predicted side matches detected side
                if cand_side == hitter_side:
                    final_stroke_en = BST_TO_ENGLISH.get(cand_stroke_cn, cand_stroke_cn)
                    break
            
            results.append([f"Shot {shot_id}", hitter_side, final_stroke_en])
            
        return results

    def _determine_hitter_side(self, hit_frame, pose_cache, shuttle_2d):
        """Identifies if the Top (Far) or Bottom (Near) player is closer to the shuttle hit."""
        if not pose_cache or hit_frame >= len(pose_cache): return "Unknown"
        pf = pose_cache[hit_frame]
        shut_pos = shuttle_2d[hit_frame]
        
        if np.isnan(shut_pos[0]): return "Bottom" # Default fallback
        
        def dist_to_shuttle(player):
            if player is None or player.bbox is None: return 1e9
            # Calculate distance between shuttle and center of player bounding box
            bx = (player.bbox[0] + player.bbox[2]) / 2
            by = (player.bbox[1] + player.bbox[3]) / 2
            return np.sqrt((bx - shut_pos[0])**2 + (by - shut_pos[1])**2)

        d_near = dist_to_shuttle(pf.near)
        d_far = dist_to_shuttle(pf.far)
        
        return "Top" if d_far < d_near else "Bottom"

    def _extract_features(self, start, end, s2d_all, pose_cache, calib, traj_3d, W, H):
        """Feature extraction logic."""
        jnb_seq, pos_seq, s5d_seq = np.zeros((100, 2, 72), dtype=np.float32), np.zeros((100, 2, 2), dtype=np.float32), np.zeros((100, 5), dtype=np.float32)
        indices = np.linspace(start, end - 1, 100).astype(int)
        for t, orig_f in enumerate(indices):
            u, v = s2d_all[orig_f]
            s2d_norm = [u/W, v/H] if not np.isnan(u) else [0, 0]
            rel_idx = orig_f - start
            s3d_norm = (np.nan_to_num(traj_3d[rel_idx]) / [COURT_W, COURT_L, COURT_H]) if (traj_3d is not None and rel_idx < len(traj_3d)) else [0, 0, 0]
            s5d_seq[t] = np.concatenate([s2d_norm, s3d_norm])
            if pose_cache and orig_f < len(pose_cache):
                pf = pose_cache[orig_f]
                for p_idx, p_attr in enumerate(['near', 'far']):
                    player = getattr(pf, p_attr)
                    if player and player.keypoints:
                        kpts = np.array(player.keypoints)[:, :2] / [W, H]
                        jnb_seq[t, p_idx, :34] = kpts.flatten()
                        bones = np.zeros((19, 2))
                        for b_idx, (j1, j2) in enumerate(COCO_BONES[:19]): bones[b_idx] = kpts[j1] - kpts[j2]
                        jnb_seq[t, p_idx, 34:72] = bones.flatten()
                        if player.floor_pos_3d: pos_seq[t, p_idx] = np.nan_to_num(player.floor_pos_3d[:2]) / [COURT_W, COURT_L]
        return jnb_seq, pos_seq, s5d_seq