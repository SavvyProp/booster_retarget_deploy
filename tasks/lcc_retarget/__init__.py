#from booster_deploy.utils.isaaclab.configclass import configclass
#from booster_deploy.utils.registry import register_task
#from .lcc_retarget import T1RetargetControllerCfg
#import os
# Register locomotion tasks


#@configclass
#class T1RetargetControllerCfg1(T1RetargetControllerCfg):
#    '''Human-like walk for T1 robot.'''
#    def __post_init__(self):
#        super().__post_init__()
#        self.policy.checkpoint_path = "tasks/pd_retarget/models/accad_W2K/policy.onnx"


#register_task(
#    "t1_lcc_retarget", T1RetargetControllerCfg1())