import pinocchio as pin
import jax.numpy as jnp
import jax
import numpy as np
from pin.local_lcc import step as lcc_step
# This class (file) defines a class used to extract key model properties using Pinocchio

class PinLCC:
    def __init__(self, urdf_path: str, mesh_dir: str):
        self.model, _, _ = pin.buildModelsFromUrdf(
            urdf_path,
            mesh_dir,
            root_joint=pin.JointModelFreeFlyer()
        )
        with np.load("pin/booster_ids.npz") as npz:
            self.bids = {k: npz[k] for k in npz}  # or: for k in npz.keys()
        self._ids = {}
        for k, v in self.bids.items():
            # Convert 0-d arrays / numpy scalars to Python scalars (static-friendly)
            if isinstance(v, np.ndarray) and v.shape == ():
                self._ids[k] = v.item()
            # Also handle numpy scalar types directly
            elif isinstance(v, np.generic):
                self._ids[k] = v.item()
            else:
                # Keep real arrays as JAX arrays
                self._ids[k] = jnp.asarray(v) if isinstance(v, np.ndarray) else v

        self.data = self.model.createData()

        self.qpin = np.zeros([self.model.nq])
        self.vpin = np.zeros([self.model.nv])
        #self.lcc_step = jax.jit(lcc_step)

        self.lcc_step = jax.jit(
            lambda base_linvel, base_angvel, grav_vac,
                   qpos, qvel, com_pos, eefpos, jacs, h, act: (
                lcc_step(base_linvel, base_angvel, grav_vac,
                         qpos, qvel, com_pos, eefpos, jacs, h, act,
                         self._ids)
            )
        )

    def updateQpin(self, joint_pos, joint_vel, 
                   base_linvel, base_angvel,
                   grav_vec):
        g = pin.Motion.Zero()
        g.linear = grav_vec

        self.qpin = np.concatenate([np.zeros(3),
                                    np.array([0.0, 0.0, 0.0, 1.0]),
                                    joint_pos], axis=-1)
        self.vpin = np.concatenate([base_linvel, base_angvel,
                                     joint_vel], axis=-1)
        pin.computeAllTerms(self.model, self.data, self.qpin, self.vpin)
        pin.forwardKinematics(self.model, self.data, self.qpin, self.vpin)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.qpin)

        eefs = ["Left_Hand_Roll", "Right_Hand_Roll",
                "Left_Ankle_Roll", "Right_Ankle_Roll"]
        self.eef_ids = []
        for eef in eefs:
            eid = self.model.getJointId(eef)
            self.eef_ids.append(eid)
        return
    
    def get_com(self):
        return np.array(pin.centerOfMass(self.model, self.data, self.qpin))
    
    def get_jacobians(self):
        jacs = np.zeros([len(self.eef_ids) * 6, self.model.nv])
        for i, eid in enumerate(self.eef_ids):
            jac = pin.getJointJacobian(self.model, self.data, eid,
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            jacs[i * 6: (i + 1) * 6, :] = np.array(jac)
        return jacs
    
    def get_eef_pos(self):
        pos = np.zeros([len(self.eef_ids), 3])
        for i, eid in enumerate(self.eef_ids):
            pos_ = self.data.oMi[eid]
            pos[i, :] = np.array(pos_.translation)
        return pos
    
    def get_grav_comp(self):
        return np.array(self.data.nle)
    
    def step(self, base_linvel, base_angvel, grav_vec,
             joint_pos, joint_vel, action):
        self.updateQpin(joint_pos, joint_vel, 
                        base_linvel, base_angvel,
                        grav_vec)
        com_pos = jnp.array(self.get_com())
        eef_pos = jnp.array(self.get_eef_pos())
        jacs = jnp.array(self.get_jacobians())
        h = jnp.array(self.get_grav_comp())

        u_ff, pd_tau, des_pos, f, nle_ff = self.lcc_step(
                        jnp.array(base_linvel), jnp.array(base_angvel), 
                        jnp.array(grav_vec),
                        jnp.array(joint_pos), 
                        jnp.array(joint_vel),
                        com_pos, eef_pos, jacs, h, 
                        jnp.array(action))
        
        return u_ff, pd_tau, des_pos, f, nle_ff
        

    def print_q_order(self) -> None:
        """Print which joint corresponds to which indices in q (configuration)."""
        print(f"model.nq={self.model.nq}, model.nv={self.model.nv}, model.njoints={self.model.njoints}")
        for jid in range(self.model.njoints):
            jname = self.model.names[jid]  # joint id -> name
            jmodel = self.model.joints[jid]
            # Each joint occupies q[jmodel.idx_q : jmodel.idx_q + jmodel.nq]
            q0 = jmodel.idx_q
            q1 = q0 + jmodel.nq
            print(f"jid={jid:2d}  name={jname:30s}  q[{q0}:{q1}] (nq_joint={jmodel.nq})")



if __name__ == "__main__":
    pin_model = PinLCC(
        urdf_path="models/booster_t1_pgnd/T1_29dof.urdf",
        mesh_dir="models/booster_t1_pgnd/"
    )
    pin_model.print_q_order()
    joint_pos = np.zeros((29,))
    joint_vel = np.zeros((29,))
    base_linvel = np.zeros((3,))
    base_angvel = np.zeros((3,))
    grav_vec = np.array([0.0, 0.0, -9.81])
    pin_model.updateQpin(joint_pos, joint_vel, base_linvel, base_angvel,
                         grav_vec)
    com = pin_model.get_com()
    print("COM:", com)
    jacs = pin_model.get_jacobians()
    print("Jacobians shape:", jacs.shape)
    eef_pos = pin_model.get_eef_pos()
    print("End-effector positions:", eef_pos)
    grav_comp = pin_model.get_grav_comp()
    print("Gravity compensation torques:", grav_comp)
    import time
    while True:
        st = time.perf_counter()
        u_ff, pd_tau, des_pos, f, nle_ff = pin_model.step(base_linvel, base_angvel, grav_vec,
                    joint_pos, joint_vel,
                    action = np.zeros((2 * 29 + 6 + 5,)))
        print("Step time:", time.perf_counter() - st)
        