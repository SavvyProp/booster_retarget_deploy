from vicon_ros.vicon_listen import ViconTFClient
from booster_deploy.utils.math import quat_mul, rotate_vector_by_quat, rotmat_to_rpy
import numpy as np
vicon_client = ViconTFClient()
offset = [ 0.16337996,  -0.06678951, -0.98429104, 0.00415838]
import time
last_time = time.time()
prev_pos = np.zeros([3])
global_vel = np.zeros([3])
while True:
    #vicon_client.print_all_frames()
    try:
        vicon_pos, vicon_quat, rpy = vicon_client.get_marker_position(
                "Booster/booster_seg"
                )
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
        #r_world_rpy = rotmat_to_rpy(R_world_body)
        #print(r_world_rpy)
        vicon_quat = quat_mul(offset, vicon_quat, ordering="wxyz", normalize=True)
        #print(vicon_quat)
        st = time.time()
        dt = st - last_time
        last_time = st
        alpha = 0.01
        
        raw_global_vel = (vicon_pos - prev_pos) / dt
        prev_pos = vicon_pos
        global_vel = global_vel * (1 - alpha) + raw_global_vel * alpha

        #vicon_quat_conj = vicon_quat * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        local_vel = np.linalg.inv(R_world_body) @ global_vel
        #local_vel = rotate_vector_by_quat(
        #    global_vel, vicon_quat_conj, ordering="wxyz")
        
        print("Local vel x: {:.3f} y: {:.3f} z: {:.3f}".format(
            local_vel[0], local_vel[1], local_vel[2]))
    except Exception as e:
        print("Failed to get marker position:", e)
        continue

    