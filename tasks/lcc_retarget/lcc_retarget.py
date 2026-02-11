from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import T1_29DOF_LCC_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
from tasks.lcc_retarget.pin.pin_lcc import PinLCC
import onnxruntime as ort
import numpy as np
from typing import List, Optional
import torch
import time as time_
from dataclasses import MISSING
import onnx

isaac_to_mj = [
    0, 4, 1, 5, 9, 13, 17, 21, 25, 2, 6, 10, 14, 18, 22, 26, 3, 7, 11, 15, 19, 23, 27, 8, 12, 16, 20, 24, 28
]
mj_ankle = [21, 22, 27, 28]
not_mj_ankle = [i for i in range(29) if i not in mj_ankle]
reorder = mj_ankle + not_mj_ankle
mj_to_isaac = [
    0, 2, 9, 16, 1, 3, 10, 17, 23, 4, 11, 18, 24, 5, 12, 19, 25, 6, 13, 20, 26, 7, 14, 21, 27, 8, 15, 22, 28
]

mj_to_isaac_ankle = [
    21, 27, 22, 28, 0, 2, 9, 16, 1, 3, 10, 17, 23, 4, 11, 18, 24, 5, 12, 19, 25, 6, 13, 20, 26, 7, 14, 8, 15
]

is_joint_pos = np.array(
    [
    0.0, 0.2, 0.2, 0.0, 0.0, -1.35, 1.35, -0.2, -0.2, 0.0,
    0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.42,
    0.42, 0.0, 0.0, -0.23, -0.23, 0.0, 0.0, 0.0, 0.0,
]
).astype(np.float32)

is_joint_pos_ankle = np.array(
    [
    -0.23, -0.23, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, -1.35, 1.35, -0.2, -0.2, 0.0,
    0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.42,
    0.42, 0.0, 0.0, 0.0, 0.0,
]
).astype(np.float32)

def quat_to_rotmat_wxyz(q):
    """q = [w, x, y, z]. Returns R such that v_world = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

class LCCRetargetPolicy(Policy):
    def __init__(self, cfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.session = ort.InferenceSession(self.cfg.checkpoint_path)
        self.last_action = np.zeros((29 * 2 + 7 + 4,), dtype=np.float32)
        self.counter = 0
        self.delay = 0
        for inp in self.session.get_inputs():
            if inp.name == "obs":
                self.obs_size = inp.shape[1]

        self.history_length = 1
        if self.obs_size > 191:
            self.history_length = 4

        self.obs_hist = None #np.zeros((self.history_length, 191), dtype=np.float32)
        
        dummy_obs = np.zeros((1, self.obs_size)).astype(np.float32)
        
        dummy_time = np.array([[self.counter - self.delay]]).astype(np.float32)
        try:
            onm = onnx.load(self.cfg.checkpoint_path)
            metadata_dict = {p.key: p.value for p in onm.metadata_props}  
            duration = int(metadata_dict["seq_len"]) - 1
            duration = min(duration, 500)
        except:
            duration = 500
        initial_out = self.session.run(None, 
                                       {"obs": dummy_obs, 
                                        "time_step": dummy_time})
        # Out index is actions, joint pos, joint vel,
        # body pos w, body quat w, body lin vel w, body ang vel w
        self.prev_joint_pos = initial_out[1]
        self.prev_joint_vel = initial_out[2]
        self.prev_body_pos = initial_out[3]
        self.prev_body_quat = initial_out[4]
        self.prev_body_vel = initial_out[5]
        self.prev_body_angvel = initial_out[6]
        self.duration = duration
        self.obs = np.zeros((self.obs_size,), dtype=np.float32)
        self.pin_lcc = PinLCC(
            urdf_path="tasks/lcc_retarget/booster_t1_pgnd/T1_29dof.urdf",
            mesh_dir="tasks/lcc_retarget/booster_t1_pgnd/",
            pin_npz_dir="tasks/lcc_retarget/pin/booster_ids.npz"
        )
        self.decimation_counter = 0
        self.f = None
        self.com_vel = None
        self.com_accs = None
        self.com_angvel = None
        self.w = None
        self.policy_last_time = time_.perf_counter()
        self.policy_dt = 0.0
        self.joint_vel = np.zeros((29,), dtype=np.float32)
        self.alpha = np.ones((29,), dtype=np.float32)
        ankle_filt = 1.0
        self.alpha[21] = ankle_filt
        self.alpha[22] = ankle_filt
        self.alpha[27] = ankle_filt
        self.alpha[28] = ankle_filt

        self.vel_limit = np.ones((29,), dtype=np.float32) * 10.0
        self.vel_limit[21] = 5.0
        self.vel_limit[22] = 5.0
        self.vel_limit[27] = 5.0
        self.vel_limit[28] = 5.0

        self.prev_base_vel = np.zeros([3,])
        self.prev_u_ff = np.zeros((29,), dtype=np.float32)
        self.prev_des_pos = np.zeros((29,), dtype=np.float32)

    def reset(self):
        self.counter = 0
        self.last_action = np.zeros_like(self.last_action)
        return
    
    def compute_observation(self, dof_pos, dof_vel, base_ang_vel, base_lin_vel):
        """Compute current observation following sim2sim.py pattern."""
        
        dof_vel = np.clip(dof_vel, -self.vel_limit, self.vel_limit)

        if self.history_length > 1:
            mapped_dof_pos = dof_pos[mj_to_isaac_ankle] - is_joint_pos_ankle
            mapped_dof_vel = dof_vel[mj_to_isaac_ankle]
        else:
            mapped_dof_pos = dof_pos[mj_to_isaac] - is_joint_pos
            mapped_dof_vel = dof_vel[mj_to_isaac]

        if self.counter < self.delay:
            self.prev_joint_pos = is_joint_pos.reshape(1, -1)
            self.prev_joint_vel = np.zeros((1, 29), dtype=np.float32)
            
        cmd = np.concatenate([
            self.prev_joint_pos,
            self.prev_joint_vel,], axis = -1).astype(np.float32)
        
        #self.prev_base_vel = base_lin_vel * 0.3 + self.prev_base_vel * 0.70

        obs = np.concatenate([
            cmd[0, :], 
            base_lin_vel,
            base_ang_vel,
            mapped_dof_pos,
            mapped_dof_vel,
            self.last_action.reshape(-1)
        ], axis=-1)

        self.obs = obs
        if self.obs_hist is None:
            self.obs_hist = np.tile(obs.reshape(1, -1), (self.history_length, 1))
        else:
            self.obs_hist[:-1, :] = self.obs_hist[1:, :]
            self.obs_hist[-1, :] = obs
        return obs
    
    def eval_network(self, obs):
        if self.history_length > 1:
            cmd = self.obs_hist[:, :29 * 2].reshape(1, -1)
            base_lin_vel = self.obs_hist[:, 29 * 2: 29 * 2 + 3].reshape(1, -1)
            base_ang_vel = self.obs_hist[:, 29 * 2 + 3: 29 * 2 + 6].reshape(1, -1)
            dof_pos = self.obs_hist[:, 29 * 2 + 6: 29 * 3 + 6]
            dof_pos_high = dof_pos[:, :4].reshape(1, -1)
            dof_pos_low = dof_pos[:, 4:].reshape(1, -1)
            dof_vel = self.obs_hist[:, 29 * 3 + 6: 29 * 4 + 6]
            dof_vel_high = dof_vel[:, :4].reshape(1, -1)
            dof_vel_low = dof_vel[:, 4:].reshape(1, -1)
            last_action = self.obs_hist[:, 29 * 4 + 6:].reshape(1, -1)
            obs = np.concatenate([cmd, base_lin_vel, base_ang_vel, dof_pos_high,
                                  dof_pos_low, dof_vel_high, dof_vel_low, last_action], axis=-1)

        time = np.array([[self.counter]]).astype(np.float32)
        time = time.reshape(1, -1)
        obs = obs.reshape(1, -1).astype(np.float32)
        output = self.session.run(None, 
                                  {"obs": obs, 
                                   "time_step": time})
        
        self.counter += 1
        
        self.counter = self.counter % (self.duration + self.delay)
        self.prev_joint_pos = output[1]
        self.prev_joint_vel = output[2]
        self.prev_body_pos = output[3]
        self.prev_body_quat = output[4]
        self.prev_body_vel = output[5]
        self.prev_body_angvel = output[6]
        action = output[0]
        self.last_action = action
        #plt = time_.perf_counter() - self.initial_time
        tt = time_.perf_counter()
        print("Policy Inference Time: {:.6f} ms".format((tt) * 1000.0))
        self.policy_dt = tt - self.policy_last_time
        self.policy_last_time = tt
    
    def inference(self):
        print("Counter: ", self.counter)
        
        #obs = self.compute_observation()
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_lin_vel = self.robot.data.root_lin_vel_b
        self.joint_vel = self.joint_vel * (1 - self.alpha) + dof_vel * self.alpha

        if self.decimation_counter % 5 == 0:
            obs = self.compute_observation(
                dof_pos,
                self.joint_vel,
                base_ang_vel,
                base_lin_vel
            )
            self.eval_network(obs)
            self.decimation_counter = 0

        grav_vec = self.robot.data.grav_vec_b
        u_ff, pd_pos = self.pin_lcc.step(
            base_lin_vel,
            base_ang_vel,
            grav_vec,
            dof_pos,
            self.joint_vel,
            self.last_action.reshape(-1)
        )
        #u_ff = u_ff * 0.0 + 4.0
        self.f = self.pin_lcc.f
        self.com_vel = self.pin_lcc.com_vel
        self.com_accs = self.pin_lcc.com_accs
        self.com_angvel = self.pin_lcc.com_angvel
        self.w = self.pin_lcc.w

        self.decimation_counter += 1
        #pd_pos = pd_pos.numpy()
        #pd_pos[0] = 0.0
        #pd_pos[1] = 0.0
        pd_pos = pd_pos.at[0:2].set(0.0)
        #u_ff = u_ff.at[21:23].set(0.0)
        #u_ff = u_ff.at[27:29].set(0.0)
        self.prev_des_pos = self.prev_des_pos * 0.5 + pd_pos * 0.5
        self.prev_u_ff = self.prev_u_ff * 0.5 + u_ff * 0.5
        return pd_pos, u_ff

@configclass
class LCCRetargetPolicyCfg(PolicyCfg):
    constructor = LCCRetargetPolicy
    checkpoint_path: str = MISSING  # type: ignore
    policy_joint_names: list[str] = MISSING  # type: ignore

@configclass
class BoosterRobotControllerCfg:
    low_state_dt: float = 0.0005
    metrics_max_events: int = 2000

@configclass
class MujocoControllerCfg:
    init_pos: List[float] = [0.0, 0.0, 0.65]
    init_quat: List[float] = [1.0, 0.0, 0.0, 0.0]
    decimation: int = 1
    # physics_dt will automatically be set by ControllerCfg
    physics_dt: float = None  # type: ignore
    log_states: Optional[str] = None
    visualize_reference_ghost: bool = False
    ghost_rgba: List[float] = [0.2, 0.8, 0.2, 0.25]

@configclass
class T1LCCRetargetControllerCfg(ControllerCfg):
    robot = T1_29DOF_LCC_CFG
    policy_dt = 0.004
    booster = BoosterRobotControllerCfg()
    mujoco = MujocoControllerCfg()
    policy = LCCRetargetPolicyCfg(
        checkpoint_path="models/HDM_W/policy.onnx",
        policy_joint_names = [       # joint order in isaacsim/isaaclab
        "AAHead_yaw",
        'Left_Shoulder_Pitch',
        'Right_Shoulder_Pitch',
        'Waist',
        "Head_pitch",
        'Left_Shoulder_Roll',
        'Right_Shoulder_Roll',
        'Left_Hip_Pitch',
        'Right_Hip_Pitch',
        'Left_Elbow_Pitch',
        'Right_Elbow_Pitch',
        'Left_Hip_Roll',
        'Right_Hip_Roll',
        'Left_Elbow_Yaw',
        'Right_Elbow_Yaw',
        'Left_Hip_Yaw',
        'Right_Hip_Yaw',
        'Left_Knee_Pitch',
        'Right_Knee_Pitch',
        'Left_Ankle_Pitch',
        'Right_Ankle_Pitch',
        'Left_Ankle_Roll',
        'Right_Ankle_Roll'
    ]
    )