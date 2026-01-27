from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .lcc_retarget import T1LCCRetargetControllerCfg
import os
# Register locomotion tasks


@configclass
class T1LCCRetargetControllerCfg1(T1LCCRetargetControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/CMU_41_02/policy.onnx"


register_task(
    "t1_lcc_retarget", T1LCCRetargetControllerCfg1())