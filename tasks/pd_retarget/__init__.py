from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .pd_retarget import T1RetargetControllerCfg
import os
# Register locomotion tasks


@configclass
class T1_accad_W2K(T1RetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/pd_retarget/models/accad_W2K/policy.onnx"

@configclass
class T1_CMU_41_02(T1RetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/pd_retarget/models/CMU_41_02/policy.onnx"

@configclass
class T1_HDM_W(T1RetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/pd_retarget/models/HDM_W/policy.onnx"

@configclass
class T1_HDM_R(T1RetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/pd_retarget/models/HDM_R/policy.onnx"

register_task(
    "t1_retarget_w2k", T1_accad_W2K())

register_task(
    "t1_retarget_cmu_41_02", T1_CMU_41_02())

register_task(
    "t1_retarget_hdm_w", T1_HDM_W())

register_task(
    "t1_retarget_hdm_r", T1_HDM_R())