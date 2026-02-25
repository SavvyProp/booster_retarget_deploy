from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .pd_retarget import T1RetargetControllerCfg
import os
# Register locomotion tasks

@configclass
class T1PD_kick(T1RetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/pd_retarget/models/t1_23_kick_trans/policy.onnx"


@configclass
class T1PD_skiing(T1RetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/pd_retarget/models/t1_23_skiing_trans/policy.onnx"

@configclass
class T1PD_hk(T1RetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/pd_retarget/models/t1_23_hk_trans/policy.onnx"


register_task(
    "t1_retarget_kick", T1PD_kick()
)

register_task(
    "t1_retarget_skiing", T1PD_skiing()
)

register_task(
    "t1_retarget_hk", T1PD_hk()
)