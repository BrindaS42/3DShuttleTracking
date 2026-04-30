# modules/trajectory_estimator.py
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class TrajectoryEstimator:
    def __init__(self):
        self.g = np.array([0, 0, -9.81]) # Physics gravity

    def _ode(self, t, state, Cd):
        vel = state[3:]
        drag = -Cd * float(np.dot(vel, vel)) * vel
        return np.concatenate([vel, self.g + drag]) # Eq. 1[cite: 24]

    def integrate(self, x0, v0, Cd, t_end, fps):
        N = max(2, int(round(t_end * fps)) + 1)
        ts = np.linspace(0.0, t_end, N)
        sol = solve_ivp(self._ode, [0.0, t_end], np.r_[x0, v0], args=(Cd,), t_eval=ts)
        return sol.y[:3].T if sol.success else np.full((N, 3), np.nan)

    def estimate_shot(self, shot_id, shuttle_2d, P, hitter_3d, receiver_3d, fps):
        """Optimizes Eq. 4 while logging performance to console[cite: 24]."""
        start_time = time.time()
        print(f"\n[Trajectory] Processing Shot {shot_id} | Frames: {len(shuttle_2d)}")
        
        N = len(shuttle_2d)
        t_total = (N - 1) / fps
        sigma = 1.0 / (np.linalg.norm(P) ** 2)

        def loss(p):
            x0, v0, Cd = p[:3], p[3:6], float(np.exp(p[6]))
            traj = self.integrate(x0, v0, Cd, t_total, fps)
            if np.any(np.isnan(traj)): return 1e9
            
            pts_homo = np.hstack([traj, np.ones((len(traj), 1))])
            proj = (P @ pts_homo.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            
            valid = ~np.any(np.isnan(shuttle_2d), axis=1)
            Lr = np.sum(np.linalg.norm(proj[valid] - shuttle_2d[valid], axis=1)**2)
            
            total_loss = (sigma * 100.0) * Lr
            if hitter_3d is not None: total_loss += np.linalg.norm(traj[0] - hitter_3d)**2
            if receiver_3d is not None: total_loss += np.linalg.norm(traj[-1] - receiver_3d)**2
            return total_loss

        p0 = np.r_[hitter_3d if hitter_3d is not None else [3, 6, 2], [0, 30, 5], np.log(0.15)]
        res = minimize(loss, p0, method="L-BFGS-B")
        
        elapsed = time.time() - start_time
        final_err = res.fun / N
        print(f"  -> Converged: {res.success} | Error: {final_err:.4f} px | Time: {elapsed:.2f}s")
        
        final_traj = self.integrate(res.x[:3], res.x[3:6], np.exp(res.x[6]), t_total, fps)
        return final_traj, final_err