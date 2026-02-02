from vicon_ros.vicon_listen import ViconTFClient
import numpy as np
import time

MARKER_OFFSET = np.array([0.13936, 0.0, 0.21265])

class ViconVelocityEstimator:
    def __init__(self):
        self.vicon_client = ViconTFClient()
        self.vicon_pos = np.zeros([3])
        self.global_vel = np.zeros([3])
        self.local_vel =  np.zeros([3])
        #self.last_time = time.time()
        self.last_time = 0.0
        self.global_ori = np.eye(3).flatten()
        #self.last_nsec = None

    def update(self):
        try:
            vicon_pos, vicon_quat, rpy, time_since = self.vicon_client.get_marker_position(
                "Booster/booster_seg"
                )
            
            print("Vicon Time:", time_since)
            
            if abs(time_since - self.last_time) < 1e-4:
                return  # Skip update if no new data

            marker_offset_body = np.array([0.150, 0.0, 0.162]) # top-center of Booster
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
            alpha = 0.2
            raw_global_vel = (vicon_pos - self.vicon_pos) / dt
            self.vicon_pos = vicon_pos
            self.global_vel = self.global_vel * (1 - alpha) + raw_global_vel * alpha

            offset = R_world_body @ MARKER_OFFSET #Rotate marker offset into world frame
            self.vicon_pos = vicon_pos - offset


            self.local_vel = np.linalg.inv(R_world_body) @ self.global_vel
            #self.logger.info("Local vel x: {:.3f} y: {:.3f} z: {:.3f}".format(
            #    self.local_vel[0], self.local_vel[1], self.local_vel[2]))
            #self.global_pos = vicon_pos
            self.global_ori = R_world_body.flatten()
        except Exception as e:
            print("Failed to get marker position:", e)

        return self.vicon_pos, self.local_vel, self.global_ori