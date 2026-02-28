from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .lcc_retarget import T1LCCRetargetControllerCfg
import os
# Register locomotion tasks


@configclass
class T1LCC_kick(T1LCCRetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/t1_23_kick_trans/policy.onnx"


@configclass
class T1LCC_skiing(T1LCCRetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/t1_23_skiing_trans/policy.onnx"

@configclass
class T1LCC_hk(T1LCCRetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/t1_23_hk_trans/policy.onnx"

@configclass
class T1LCC_balance(T1LCCRetargetControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/t1_23_balance_trans/policy.onnx"


register_task(
    "t1_lcc_retarget_kick", T1LCC_kick()
)

register_task(
    "t1_lcc_retarget_skiing", T1LCC_skiing()
)

register_task(
    "t1_lcc_retarget_hk", T1LCC_hk()
)

register_task(
    "t1_lcc_retarget_balance", T1LCC_balance()
)