from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .lcc_retarget import T1LCCRetargetControllerCfg
import os
# Register locomotion tasks


@configclass
class T1LCC_CMU_41_02(T1LCCRetargetControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/CMU_41_02/policy.onnx"

@configclass
class T1LCC_HDM_D(T1LCCRetargetControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/HDM_D/policy.onnx"

@configclass
class T1LCC_HDM_WS(T1LCCRetargetControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/HDM_WS/policy.onnx"

@configclass
class T1LCC_lafan_sidesteps(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/lafan_sidesteps/policy.onnx"

@configclass
class T1LCC_lafan_sidesteps_2(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/lafan_sidesteps_2/policy.onnx"

@configclass
class T1LCC_accad_W2K(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/accad_W2K/policy.onnx"

@configclass
class T1LCC_standing(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/standing/policy.onnx"

@configclass
class T1LCC_HDM_WT(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/HDM_WT/policy.onnx"

@configclass
class T1LCC_HDM_WT_rough(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/HDM_WT_rough/policy.onnx"

@configclass
class T1LCC_HDM_WT_rand(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/HDM_WT_rand/policy.onnx"

@configclass
class T1LCC_HDM_WT_2(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/HDM_WT_2/policy.onnx"

@configclass
class T1LCC_CMU_sidestep(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/CMU_sidestep/policy.onnx"

@configclass
class T1LCC_CMU_backstep(T1LCCRetargetControllerCfg):
    '''Sidestepping motion for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "tasks/lcc_retarget/models/CMU_backstep/policy.onnx"

register_task(
    "t1_lcc_retarget_CMU_41_02", T1LCC_CMU_41_02())

register_task(
    "t1_lcc_retarget_HDM_D", T1LCC_HDM_D())

register_task(
    "t1_lcc_retarget_HDM_WS", T1LCC_HDM_WS())

register_task(
    "t1_lcc_retarget_lafan_sidesteps", T1LCC_lafan_sidesteps())

register_task(
    "t1_lcc_retarget_accad_W2K", T1LCC_accad_W2K())

register_task(
    "t1_lcc_retarget_standing", T1LCC_standing()
)

register_task(
    "t1_lcc_retarget_HDM_WT", T1LCC_HDM_WT()
)

register_task(
    "t1_lcc_retarget_HDM_WT_rough", T1LCC_HDM_WT_rough()
)

register_task(
    "t1_lcc_retarget_HDM_WT_rand", T1LCC_HDM_WT_rand()
)

register_task(
    "t1_lcc_retarget_HDM_WT_2", T1LCC_HDM_WT_2()
)

register_task(
    "t1_lcc_retarget_lafan_sidesteps_2", T1LCC_lafan_sidesteps_2()
)

register_task(
    "t1_lcc_retarget_CMU_sidestep", T1LCC_CMU_sidestep()
)

register_task(
    "t1_lcc_retarget_CMU_backstep", T1LCC_CMU_backstep()
)