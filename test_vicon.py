from vicon_ros.vicon_listen import ViconTFClient
from booster_deploy.utils.math import quat_mul, rotate_vector_by_quat, rotmat_to_rpy
import numpy as np
from booster_deploy.utils.vicon_vel import ViconVelocityEstimator
import time
vve = ViconVelocityEstimator()

local_vel_arr = np.zeros([10000, 6])

for c in range(10000):
    pos, local_vel, global_ori = vve.update()
    local_vel_arr[c, :3] = local_vel
    local_vel_arr[c, 3:] = pos
    time.sleep(0.001)
    print("Local Vel:", local_vel)
np.savetxt("eval_data/vicon_local_vel.csv", local_vel_arr, delimiter = ',')