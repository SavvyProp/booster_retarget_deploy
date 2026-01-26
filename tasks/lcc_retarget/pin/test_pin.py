import pinocchio as pin

model, _, _ = pin.buildModelsFromUrdf("models/booster_t1_pgnd/T1_29dof.urdf",
                                    "models/booster_t1_pgnd/",
                                    root_joint = pin.JointModelFreeFlyer())