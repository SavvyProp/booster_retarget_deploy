from vicon_ros.vicon_listen import ViconTFClient
import numpy as np
import time

MARKER_OFFSET = np.array([0.13936, 0.0, 0.21265])


def rotmat_to_rpy(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to roll/pitch/yaw (radians).

    Convention matches this module's construction: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    (intrinsic XYZ / extrinsic ZYX). Returns np.array([roll, pitch, yaw]).

    Args:
        R: 3x3 rotation matrix.

    Returns:
        np.ndarray: [roll, pitch, yaw] in radians.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got shape {R.shape}")

    # Clamp for numerical safety
    r20 = float(R[2, 0])
    r20 = np.clip(r20, -1.0, 1.0)

    # For ZYX: pitch = atan2(-r20, sqrt(r00^2 + r10^2))
    pitch = np.arctan2(-r20, np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))

    # Detect gimbal lock when cos(pitch) ~ 0
    if np.isclose(np.cos(pitch), 0.0, atol=1e-8):
        # When pitch is +/-90deg, yaw and roll are coupled; set roll=0 and compute yaw.
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


class ViconVelocityEstimator:
    def __init__(self):
        self.vicon_client = ViconTFClient()
        self.vicon_pos = np.zeros([3])
        self.base_pos = np.zeros([3])
        self.global_vel = np.zeros([3])
        self.local_vel =  np.zeros([3])
        #self.last_time = time.time()
        self.last_time = None
        self.global_ori = np.eye(3).flatten()
        #self.last_nsec = None

    def update(self):
        try:
            vicon_pos, vicon_quat, rpy, time_since = self.vicon_client.get_marker_position(
                "Booster/booster_seg"
                )
            
            #print("Vicon Time:", time_since)
            if self.last_time is None:
                self.last_time = time_since
                self.vicon_pos = vicon_pos
                return self.base_pos, self.local_vel, self.global_ori
            
            if abs(time_since - self.last_time) < 1e-5:
                return self.base_pos, self.local_vel, self.global_ori  # Skip update if no new data

            marker_offset_body = np.array([0.1394, 0.0, 0.2126]) # top-center of Booster
            R_meas_body = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0,-1.0, 0.0]])

            # Additional fixed pitch tilt of the marker plane by +6 deg about BODY Y (tilt defined in true body frame)
            theta = np.deg2rad(18.36)
            R_body_markers = np.array([[ np.cos(theta), 0.0, np.sin(theta)],
                                        [ 0.0,          1.0, 0.0         ],
                                        [-np.sin(theta), 0.0, np.cos(theta)]])

            # Total body->measured mapping including mounting tilt (apply body tilt first, then body->measured axis mapping)
            marker_offset_meas = R_meas_body @ (R_body_markers @ marker_offset_body)
            cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
            cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
            cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
            R_x = np.array([[1.0, 0.0, 0.0],
                                [0.0,  cr, -sr],
                                [0.0,  sr,  cr]])
            R_y = np.array([[ cp, 0.0, sp],
                                [0.0, 1.0, 0.0],
                                [-sp, 0.0, cp]])
            R_z = np.array([[ cy, -sy, 0.0],
                                [ sy,  cy, 0.0],
                                [0.0, 0.0, 1.0]])
            R_world_meas = R_z @ R_y @ R_x
            R_world_body = R_world_meas @ (R_meas_body @ R_body_markers)

            dt = time_since - self.last_time
            self.last_time = time_since
            alpha = 0.50
            vicon_pos = vicon_pos - marker_offset_meas
            raw_global_vel = (vicon_pos - self.vicon_pos) / dt
            self.vicon_pos = vicon_pos
            self.global_vel = self.global_vel * (1 - alpha) + raw_global_vel * alpha

            offset = R_world_body @ MARKER_OFFSET #Rotate marker offset into world frame
            self.base_pos = vicon_pos - offset

            print("Rotation Matrix:\n", R_world_body)
            print("RPY:", rotmat_to_rpy(R_world_body))


            self.local_vel = np.linalg.inv(R_world_body) @ self.global_vel
            #self.logger.info("Local vel x: {:.3f} y: {:.3f} z: {:.3f}".format(
            #    self.local_vel[0], self.local_vel[1], self.local_vel[2]))
            #self.global_pos = vicon_pos
            self.global_ori = R_world_body.flatten()
        except Exception as e:
            print("Failed to get marker position:", e)
        return self.base_pos, self.local_vel, self.global_ori