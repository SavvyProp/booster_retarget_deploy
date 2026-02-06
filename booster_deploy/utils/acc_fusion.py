import numpy as np

class AccelerationFusion: # Jank acceleration + Vicon velocity fusion
    def __init__(self, window_size=1000):
        self.local_vel = np.zeros([window_size, 3])

        self.acc_data = np.zeros([window_size, 3])
        self.timestamps = np.zeros([window_size])
        self.start_index = 0

        # Velocity estimates: local_vel + ∫ a dt (computed over the stored window)
        self.vel_est = np.zeros([window_size, 3])

        self.weight = np.square(np.linspace(1.0, 0.1, window_size))
        self.weight = self.weight[:, None]  # Make it a column vector
        self.last_grav_vec = np.array([0.0, 0.0, -9.81], dtype=float)

    def update(self, vicon_local_vel, imu_acc, timestamp):
        # Update circular buffer and compute new velocity.
        # Put vicon local vel into top compute smoothed vel
        # replace vicon local vel
        self.local_vel = np.roll(self.local_vel, 1, axis=0)
        self.local_vel[0, :] = vicon_local_vel
        self.acc_data = np.roll(self.acc_data, 1, axis=0)
        self.acc_data[0, :] = imu_acc - self.last_grav_vec
        self.timestamps = np.roll(self.timestamps, 1)
        self.timestamps[0] = timestamp
        self.start_index = min(self.start_index + 1, self.local_vel.shape[0])

        # For each element from 0 to start index, compute the velocity by integrating acc
        n = self.start_index
        if n < 2:
            self.vel_est[0, :] = self.local_vel[0, :]
        else:
            # Compute dt between samples (newest at index 0). Clamp non-positive/invalid dt.
            t = self.timestamps[:n]
            dt = t[:-1] - t[1:]
            dt = np.where(np.isfinite(dt) & (dt > 0.0), dt, 0.0)

            # Cumulative integral of acceleration from newest backwards: dv[i] = sum_{k< i} a[k]*dt[k]
            a = self.acc_data[:n]
            dv = np.zeros((n, 3), dtype=float)
            dv[1:, :] = np.cumsum(a[:-1, :] * dt[:, None], axis=0)

            # Velocity estimate for each stored sample
            self.vel_est[:n, :] = self.local_vel[:n, :] + dv

        # Weigh each velocity estimate: newer samples get higher weight
        vel_final = np.sum(self.weight[:n, :] * self.vel_est[:n, :], axis = 0) / np.sum(self.weight[:n, :])
        self.local_vel[0, :] = vel_final
        return vel_final
    
    def compute_grav_vec(self, acc, rpy, rotmat, g=9.80665):
        """Estimate gravity vector in the sensor/body frame.

        Uses the provided roll/pitch/yaw to rotate the world gravity vector into the
        body frame (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)). Optionally blends this
        orientation-based estimate with the accelerometer direction (helpful if the
        attitude estimate is noisy and the platform is quasi-static).

        Args:
            acc: 3-vector accelerometer measurement (m/s^2).
            rpy: 3-vector [roll, pitch, yaw] in radians.
            g: gravity magnitude.
            accel_blend: 0..1 blend factor toward accelerometer-direction estimate.
                         0 -> pure rpy-based, 1 -> pure accel-direction.

        Returns:
            3-vector gravity estimate in body frame (m/s^2).
        """
        rotmat = np.asarray(rotmat, dtype=float).reshape(3,3)
        grav_vec_vicon = rotmat.T @ np.array([0.0, 0.0, -9.81], dtype=np.float32)
        acc = np.asarray(acc, dtype=float).reshape(3,)
        rpy = np.asarray(rpy, dtype=float).reshape(3,)

        roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0,  cr, -sr],
                        [0.0,  sr,  cr]], dtype=float)
        R_y = np.array([[ cp, 0.0, sp],
                        [0.0, 1.0, 0.0],
                        [-sp, 0.0, cp]], dtype=float)
        R_z = np.array([[ cy, -sy, 0.0],
                        [ sy,  cy, 0.0],
                        [0.0, 0.0, 1.0]], dtype=float)

        # World->body rotation in the same convention as elsewhere: R_world_body = Rz @ Ry @ Rx
        # So body gravity is: g_body = R_world_body.T @ g_world
        R_world_body = R_z @ R_y @ R_x
        g_world = np.array([0.0, 0.0, -g], dtype=float)
        g_body_from_rpy = (R_world_body.T @ g_world)

        # Accelerometer-based direction (quasi-static assumption). Use only direction.
        acc_norm = float(np.linalg.norm(acc))
        if np.isfinite(acc_norm) and acc_norm > 1e-6:
            g_body_from_acc_dir = g * (acc / acc_norm)
        else:
            g_body_from_acc_dir = g_body_from_rpy

        rpy_weight = 0.0
        vicon_weight = 0.8
        grav_vec_weight = 0.2
        g_body = rpy_weight * g_body_from_rpy + vicon_weight * grav_vec_vicon + grav_vec_weight * g_body_from_acc_dir

        g_body = g_body * 9.81 / np.linalg.norm(g_body)
        alpha = 0.5
        print(g_body_from_rpy, grav_vec_vicon, g_body_from_acc_dir)
        self.last_grav_vec = g_body * alpha + (1.0 - alpha) * self.last_grav_vec

        return self.last_grav_vec
