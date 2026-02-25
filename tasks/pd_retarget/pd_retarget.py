from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.onnx_runtime import create_inference_session
import numpy as np
import onnx
from booster_deploy.utils.consts import BOOSTER_CONSTS

from dataclasses import MISSING

CTRL_NUM = 23

class PDRetargetPolicy(Policy):
    def __init__(self, cfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.session = create_inference_session(
            self.cfg.checkpoint_path,
            device=str(self.cfg.device),
            prefer_gpu=self.cfg.prefer_gpu,
            cuda_device_id=self.cfg.cuda_device_id,
            intra_op_num_threads=self.cfg.intra_op_num_threads,
            inter_op_num_threads=self.cfg.inter_op_num_threads,
        )
        if self.cfg.prefer_gpu and "CUDAExecutionProvider" not in self.session.get_providers():
            print("CUDAExecutionProvider not available. Falling back to CPUExecutionProvider.")
        print(f"ONNX Runtime providers: {self.session.get_providers()}")
        self.last_action = np.zeros((CTRL_NUM), dtype=np.float32)
        self.counter = 0
        self.delay = 0
        for inp in self.session.get_inputs():
            if inp.name == "obs":
                self.obs_size = inp.shape[1]

        self.history_length = 1
        if self.obs_size > 151:
            self.history_length = 4

        self.vel_limit = np.ones((CTRL_NUM,), dtype=np.float32) * 10.0
        
        self.obs_hist = None
        
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

    def reset(self):
        self.counter = 0
        self.last_action = np.zeros_like(self.last_action)
        return
    
    def compute_observation(self, dof_pos, dof_vel, base_ang_vel, base_lin_vel):
        """Compute current observation following sim2sim.py pattern."""
        
        dof_vel = np.clip(dof_vel, -self.vel_limit, self.vel_limit)

        if self.history_length > 1:
            mapped_dof_pos = dof_pos[BOOSTER_CONSTS.mj_to_isaac_ankle] - BOOSTER_CONSTS.is_joint_pos_ankle.astype(np.float32)
            mapped_dof_vel = dof_vel[BOOSTER_CONSTS.mj_to_isaac_ankle]
        else:
            mapped_dof_pos = dof_pos[BOOSTER_CONSTS.mj_to_isaac] - BOOSTER_CONSTS.is_joint_pos.astype(np.float32)
            mapped_dof_vel = dof_vel[BOOSTER_CONSTS.mj_to_isaac]

        if self.counter < self.delay:
            if self.history_length > 1:
                self.prev_joint_pos = BOOSTER_CONSTS.is_joint_pos_ankle.reshape(1, -1)
            else:
                self.prev_joint_pos = BOOSTER_CONSTS.is_joint_pos.reshape(1, -1)
            self.prev_joint_vel = np.zeros((1, CTRL_NUM), dtype=np.float32)
            
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
    
    def inference(self):
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_lin_vel = self.robot.data.root_lin_vel_b
        obs = self.compute_observation(dof_pos, dof_vel, base_ang_vel, base_lin_vel)
        time = np.array([[self.counter]]).astype(np.float32)
        time = time.reshape(1, -1)
        obs = obs.reshape(1, -1).astype(np.float32)
        if self.history_length > 1:
            cmd = self.obs_hist[:, :CTRL_NUM * 2].reshape(1, -1)
            base_lin_vel = self.obs_hist[:, CTRL_NUM * 2: CTRL_NUM * 2 + 3].reshape(1, -1)
            base_ang_vel = self.obs_hist[:, CTRL_NUM * 2 + 3: CTRL_NUM * 2 + 6].reshape(1, -1)
            dof_pos = self.obs_hist[:, CTRL_NUM * 2 + 6: CTRL_NUM * 3 + 6]
            dof_pos_high = dof_pos[:, :4].reshape(1, -1)
            dof_pos_low = dof_pos[:, 4:].reshape(1, -1)
            dof_vel = self.obs_hist[:, CTRL_NUM * 3 + 6: CTRL_NUM * 4 + 6]
            dof_vel_high = dof_vel[:, :4].reshape(1, -1)
            dof_vel_low = dof_vel[:, 4:].reshape(1, -1)
            last_action = self.obs_hist[:, CTRL_NUM * 4 + 6:].reshape(1, -1)
            obs = np.concatenate([cmd, base_lin_vel, base_ang_vel, dof_pos_high,
                                  dof_pos_low, dof_vel_high, dof_vel_low, last_action], axis=-1)
        
        obs = obs.astype(np.float32)
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
        joint_pos_target = action[:, BOOSTER_CONSTS.isaac_to_mj] * BOOSTER_CONSTS.action_scale
        #offset = np.array(self.robot.default_joint_pos)
        #joint_pos_target = joint_pos_target.reshape(-1) + offset.reshape(-1)
        joint_pos_target = joint_pos_target.reshape(-1) + BOOSTER_CONSTS.is_joint_pos[BOOSTER_CONSTS.isaac_to_mj].reshape(-1)
        joint_pos_target[0] = 0.0
        joint_pos_target[1] = 0.0
        u_ff = np.zeros_like(joint_pos_target)
        return joint_pos_target, u_ff

@configclass
class PDRetargetPolicyCfg(PolicyCfg):
    constructor = PDRetargetPolicy
    checkpoint_path: str = MISSING  # type: ignore
    policy_joint_names: list[str] = MISSING  # type: ignore
    prefer_gpu: bool = True
    cuda_device_id: int = 0
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0

@configclass
class T1RetargetControllerCfg(ControllerCfg):
    robot = T1_23DOF_CFG
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
