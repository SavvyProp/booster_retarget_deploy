from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import K1_CFG, T1_29DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
import onnxruntime as ort
import numpy as np
import torch

from dataclasses import MISSING

isaac_to_mj = [
    0, 4, 1, 5, 9, 13, 17, 21, 25, 2, 6, 10, 14, 18, 22, 26, 3, 7, 11, 15, 19, 23, 27, 8, 12, 16, 20, 24, 28
]
mj_to_isaac = [
    0, 2, 9, 16, 1, 3, 10, 17, 23, 4, 11, 18, 24, 5, 12, 19, 25, 6, 13, 20, 26, 7, 14, 21, 27, 8, 15, 22, 28
]

action_scale = np.array(
    [
    0.12665148, 0.12665148, 0.22797266, 0.22797266, 0.22797266, 0.22797266,
    0.22797266, 0.22797266, 0.22797266, 0.22797266, 0.22797266, 0.22797266,
    0.22797266, 0.22797266, 0.22797266, 0.22797266, 0.18997722, 0.11398633,
    0.18997722, 0.18997722, 0.15198178, 0.30396355, 0.18997722, 0.11398633,
    0.18997722, 0.18997722, 0.15198178, 0.30396355, 0.18997722
]
).astype(np.float32)

is_joint_pos = np.array(
    [
    0.0, 0.2, 0.2, 0.0, 0.0, -1.35, 1.35, -0.2, -0.2, 0.0,
    0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.42,
    0.42, 0.0, 0.0, -0.23, -0.23, 0.0, 0.0, 0.0, 0.0,
]
).astype(np.float32)


class PDRetargetPolicy(Policy):
    def __init__(self, cfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.session = ort.InferenceSession(self.cfg.checkpoint_path)
        self.last_action = np.zeros((29,), dtype=np.float32)
        self.counter = 0
        for inp in self.session.get_inputs():
            if inp.name == "obs":
                self.obs_size = inp.shape[1]
        
        dummy_obs = np.zeros((1, self.obs_size)).astype(np.float32)
        
        dummy_time = np.array([[self.counter]]).astype(np.float32)
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
        self.duration = 500
        self.obs = np.zeros((self.obs_size,), dtype=np.float32)

    def reset(self):
        self.counter = 0
        return
    
    def compute_observation(self):
        """Compute current observation following sim2sim.py pattern."""
        # Get robot state
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_lin_vel = self.robot.data.root_lin_vel_b
        # Project gravity vector into base frame
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        if self.cfg.enable_safety_fallback:
            # fall detection: stop if falling
            if projected_gravity[2] > -0.5:
                print("\nFalling detected, stopping policy for safety. "
                      "You can disable safety fallback by setting "
                      f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                      "to False.")
                #self.controller.stop()

        #default_joint_pos_sim = self.robot.default_joint_pos
        mapped_dof_pos = dof_pos[mj_to_isaac] - is_joint_pos
        mapped_dof_vel = dof_vel[mj_to_isaac] - is_joint_pos

        # Build observation: [
        #   ang_vel(3),
        #   projected_gravity(3),
        #   commands(3),
        #   joint_pos(num_action),
        #   joint_vel(num_action),
        #   actions(num_action)]

        cmd = np.concatenate([
            self.prev_joint_pos,
            self.prev_joint_vel,], axis = -1).astype(np.float32)
        
        obs = np.concatenate([
            cmd[0, :], 
            base_lin_vel.numpy(),
            base_ang_vel.numpy(),
            mapped_dof_pos.numpy(),
            mapped_dof_vel.numpy(),
            self.last_action.reshape(-1)
        ], axis=-1)

        self.obs = obs

        return obs
    
    def inference(self):
        obs = self.compute_observation()
        time = np.array([[self.counter]]).astype(np.float32)
        time = time.reshape(1, -1)
        obs = obs.reshape(1, -1).astype(np.float32)

        output = self.session.run(None, 
                                  {"obs": obs, 
                                   "time_step": time})
        
        self.counter += 1
        self.counter = self.counter % self.duration
        self.prev_joint_pos = output[1]
        self.prev_joint_vel = output[2]
        self.prev_body_pos = output[3]
        self.prev_body_quat = output[4]
        self.prev_body_vel = output[5]
        self.prev_body_angvel = output[6]
        action = output[0]
        self.last_action = action
        joint_pos_target = action[:, isaac_to_mj] * action_scale
        offset = np.array(self.robot.default_joint_pos)
        joint_pos_target = joint_pos_target.reshape(-1) + offset.reshape(-1)
        return joint_pos_target

@configclass
class PDRetargetPolicyCfg(PolicyCfg):
    constructor = PDRetargetPolicy
    checkpoint_path: str = MISSING  # type: ignore
    actor_obs_history_length: int = 10
    action_scale: float = 0.25
    obs_dof_vel_scale: float = 1.0
    policy_joint_names: list[str] = MISSING  # type: ignore

@configclass
class T1RetargetControllerCfg(ControllerCfg):
    robot = T1_29DOF_CFG
    policy = PDRetargetPolicyCfg(
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