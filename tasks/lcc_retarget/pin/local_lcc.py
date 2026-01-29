import jax.numpy as jnp
import jax

GAIN_FAC = 0.75

# Collection of LCC functions to run the controller in local space

def schur_solve(qp_q, qp_c, cons_lhs, cons_rhs, reg: float = 1e-6):
    """Solve equality-constrained QP KKT system via Schur complement.

    Solves for x in:
        [Q  A^T] [x] = [c]
        [A   0 ] [λ]   [b]

    Args:
        qp_q:     (..., F, F)
        qp_c:     (..., F)
        cons_lhs: (..., M, F)  (A)
        cons_rhs: (..., M)     (b)
        reg: Optional Tikhonov regularization on Q (adds reg*I).

    Returns:
        x: (..., F)
    """
    squeeze_out = False
    if qp_q.ndim == 2:
        qp_q = qp_q[None, ...]
        qp_c = qp_c[None, ...]
        cons_lhs = cons_lhs[None, ...]
        cons_rhs = cons_rhs[None, ...]
        squeeze_out = True

    # Symmetrize Q
    Q = 0.5 * (qp_q + jnp.swapaxes(qp_q, -1, -2))

    F = Q.shape[-1]
    batch_shape = Q.shape[:-2]

    if reg and reg > 0.0:
        I = jnp.broadcast_to(jnp.eye(F, dtype=Q.dtype), (*batch_shape, F, F))
        Q = Q + reg * I

    A = cons_lhs
    AT = jnp.swapaxes(A, -1, -2)
    c = qp_c
    b = cons_rhs

    # Solve Q X = B for multiple RHS using jnp.linalg.solve (batched).
    def solve_Q(B):
        return jnp.linalg.solve(Q, B)

    # Q^{-1} A^T and Q^{-1} c
    Qinv_AT = solve_Q(AT)  # (..., F, M)
    Qinv_c = solve_Q(c[..., None])[..., 0]  # (..., F)

    # Schur system: S λ = A Q^{-1} c - b, where S = A Q^{-1} A^T
    S = A @ Qinv_AT  # (..., M, M)
    rhs_lam = (A @ Qinv_c[..., None])[..., 0] - b  # (..., M)

    lam = jnp.linalg.solve(S, rhs_lam[..., None])[..., 0]  # (..., M)

    # Recover x = Q^{-1}(c - A^T λ)
    x = solve_Q((c - (AT @ lam[..., None])[..., 0])[..., None])[..., 0]  # (..., F)

    if squeeze_out:
        x = x[0]
    return x

@jax.jit
def skew(a: jnp.ndarray) -> jnp.ndarray:
    """Return the 3x3 skew-symmetric matrix of a vector a."""
    ax, ay, az = a
    return jnp.array([
        [0,   -az,  ay],
        [az,   0,  -ax],
        [-ay,  ax,   0]
    ])

def make_centroidal_a(eefpos, com_pos, grav_vec, ids):
    m_ang = ids["angular_inertia"]
    # rotate m_ang to world frame
    #rot_mat = lmath.quat_to_rotmat(base_quat)
    #m_ang = rot_mat @ m_ang @ rot_mat.T
    anginv = jnp.linalg.inv(m_ang)
    mass = ids["mass"]
    r = eefpos - com_pos[None, :]
    f_blocks = []
    for i in range(ids["eef_num"] + 1):
        r_skew = skew(r[i, :])
        f_block = jnp.block([
            [jnp.eye(3) / mass, jnp.zeros([3, 3])],
            [anginv @ r_skew, anginv]
        ])
        f_blocks.append(f_block)
    a = jnp.concatenate(f_blocks, axis = 1)
    g = jnp.concatenate([grav_vec, jnp.array([0, 0, 0])], axis=0)
    return a, g

# Costs

def f_mag_q(w, ids):
    """Build diagonal Q for force magnitudes from flat w (E,) in JAX.

    Matches the provided PyTorch behavior:
      - logits = -clip(w, -6, 6)
      - linear scale = exp(logits)
      - angular scale = linear scale * 60.0
      - per-effector 6-tuple = [lin, lin, lin, ang, ang, ang]
      - flattened into a diagonal matrix.
    """
    # w is (E,) where E = ids["eef_num"] + 1
    # Clip and exponentiate
    logits = -jnp.clip(w, -10.0, 10.0)          # (E,)
    scale_lin = jnp.exp(logits)               # (E,)
    scale_ang = scale_lin * 10.0              # (E,)

    # Build per-effector 6-tuple: [lin, lin, lin, ang, ang, ang]
    lin3 = jnp.repeat(scale_lin[:, None], 3, axis=1)  # (E,3)
    ang3 = jnp.repeat(scale_ang[:, None], 3, axis=1)  # (E,3)
    per_eff = jnp.concatenate([lin3, ang3], axis=1)   # (E,6)

    # Flatten effector+axis to (E*6,) and build diagonal matrix
    diag_vec = per_eff.reshape(-1)                    # (E*6,)
    big_qp = jnp.diag(diag_vec)                       # (E*6, E*6)
    return big_qp, jnp.zeros(6 * (ids["eef_num"] + 1))

def joint_torque_q(jacs, tau_ref, w_diag):
    """Build QP cost terms from joint-space Jacobian and reference torques.

    Args:
        jacs:    (6*EEF_NUM, 6+CTRL_NUM) Jacobian, including 6 base dofs.
        tau_ref: (CTRL_NUM,) reference joint torques.

    Returns:
        big_q:   (6*EEF_NUM, 6*EEF_NUM) = J_j @ J_j^T
        small_q: (6*EEF_NUM,)          = J_j @ tau_ref
        where J_j = -jacs[:, 6:]  (exclude the 6 base dofs).
    """
    # Basic rank checks
    if jacs.ndim != 2:
        raise ValueError(f"jacs must be 2D (F, 6+CTRL), got shape {jacs.shape}")
    if tau_ref.ndim != 1:
        raise ValueError(f"tau_ref must be 1D (CTRL,), got shape {tau_ref.shape}")

    F, cols = jacs.shape
    if cols < 7:
        raise ValueError(f"jacs last dim must be at least 7 (6 base + >=1 ctrl), got {cols}")

    # Extract joint part J_j = -jacs[:, 6:]


    
    J_j = -jacs[:, 6:]
    ctrl_dim = J_j.shape[1]

    if tau_ref.shape[0] != ctrl_dim:
        raise ValueError(
            f"CTRL dim mismatch: J_j has {ctrl_dim} but tau_ref has {tau_ref.shape[0]}"
        )

    # big_q: (F, F) = J_j @ J_j^T
    big_q = (J_j * w_diag[None, :]) @ J_j.T

    # small_q: (F,) = J_j @ tau_ref
    small_q = J_j @ (tau_ref * w_diag)

    return big_q, small_q

# Constraints

def centroidal_qacc_cons(big_a, g, com_ref):
    # big_a @ F + g = com_ref
    lhs = big_a
    rhs = com_ref - g
    return lhs, rhs
# QP Solver


def ft_ref(eefpos, com_pos, 
         jacs, tau_ref, com_ref, w, grav_vac, ids):
    weights = jnp.array([1e-3, 1e1])
    # Should be 1e0, 1e0

    unaccounted_jacs = jnp.zeros([6, ids["ctrl_num"] + 6])
    unaccounted_jacs = unaccounted_jacs.at[:6, :6].set(jnp.eye(6))
    jacs = jnp.concatenate([unaccounted_jacs, jacs], axis = 0)
    eefpos = jnp.concatenate([com_pos[None, :], eefpos], axis = 0)

    
    a, g = make_centroidal_a(
                          eefpos,
                          com_pos,
                          grav_vac,
                          ids
                          )
    
    # Make Costs

    w_diag = jnp.square(1.0 / ids["tau_limits"])

    big_q_mag, small_c_mag = f_mag_q(w, ids)
    big_q_mag *= weights[0]
    small_c_mag *= weights[0]
    qp_q = big_q_mag
    qp_c = small_c_mag

    big_q_tau, small_c_tau = joint_torque_q(jacs, tau_ref, w_diag)
    big_q_tau *= weights[1]
    small_c_tau *= weights[1]

    qp_q += big_q_tau
    qp_c += small_c_tau

    centroid_lhs, centroid_rhs = centroidal_qacc_cons(a, g, com_ref)

    sol = schur_solve(qp_q, qp_c, centroid_lhs, centroid_rhs)

    f = sol
    tau = -jacs[:, 6:].T @ f
    debug_dict = {}

    # Test the final qp weights

    #f_mag_q_weight = f.T @ (big_q_mag @ f) * weights[0]
    #jtq_vec = tau - tau_ref
    #jt_mag = jnp.sum(jtq_vec * jtq_vec * w_diag) * weights[1]

    return tau, f, debug_dict

def make_filt_state(ids):
    state = {
        "prev_u": jnp.zeros([ids["ctrl_num"]]),
    }
    return state

def ctrl2logits(act, ids):
    c_ = ids["ctrl_num"]
    e_ = ids["eef_num"]
    des_pos = act[0:c_]
    des_com_vel = act[c_:c_ + 3]
    w = act[c_ + 3: c_ + 4 + e_]
    torque = act[c_ + e_ + 4: c_ * 2 + e_ + 4]
    des_com_angvel = act[c_ * 2 + e_ + 4: c_ * 2 + e_ + 7]
    logits = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "w": w,
        "torque": torque,
    }
    return logits

def ctrl2components(act, ids):
    logits = ctrl2logits(act, ids)
    des_pos = logits["des_pos"][ids["isaac_to_mj"]] * ids["lg_action_scale"]
    offset = ids["default_joint_pos"][ids["isaac_to_mj"]]
    
    #des_pos = des_pos + offset
    des_pos = offset
    des_angvel = logits["des_com_angvel"] * 0.50
    des_com_vel = logits["des_com_vel"] * 0.25
    
    w = logits["w"]

    torque_logit = jnp.tanh(logits["torque"][ids["isaac_to_mj"]] * 0.5)
    tau_limits = ids["tau_limits"]
    tau_naive = tau_limits * torque_logit
    tau = tau_naive
    outputs = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_angvel,
        "w": w,
        "torque": tau,
    }
    return outputs

def highlvlPD(com_vel, com_angvel,
              des_com_vel, des_angvel):
    c_lin_p_gain = 15.0
    com_acc = c_lin_p_gain * (des_com_vel - com_vel)
    c_ang_p_gain = 10.0
    com_angacc = c_ang_p_gain * (des_angvel - com_angvel)
    com_accs = jnp.concatenate([com_acc, com_angacc], axis = 0)
    return com_accs

def step(base_linvel, base_angvel, grav_vac,
         qpos, qvel, 
         com_pos, eefpos, jacs, h, 
         act, ids):
    output = ctrl2components(act, ids)
    des_pos = output["des_pos"]
    des_com_vel = output["des_com_vel"]
    des_angvel = output["des_com_angvel"]
    w = output["w"]
    tau = output["torque"]


    p_weight = ids["p_gains"]
    d_weight = ids["d_gains"]


    #com_vel = data.qvel[ids["joint_vel_ids"]][0:3]
    
    com_accs = highlvlPD(base_linvel, base_angvel,
                        des_com_vel, des_angvel)
    
    
    #s = jnp.where(nn.sigmoid(w) > 0.5, 1.0, 0.0)
    u_ff, f, norm_dict = ft_ref(
        eefpos, com_pos, jacs, tau, com_accs, w, grav_vac, ids
    )

    nle_ff = h[6:]
    u_ff = u_ff + nle_ff
    u_ff = jnp.clip(u_ff, -ids["tau_limits"] * 1.0, ids["tau_limits"] * 1.0)
    #u_ff = u_ff + nle_ff

    #p_weight = p_weight * (1.0 - torque_fac * 0.5)
    
    #pd_tau = p_weight * GAIN_FAC * (des_pos - qpos) + d_weight * (0 - qvel)
    p_tau = p_weight * GAIN_FAC * (des_pos - qpos)
    d_tau = d_weight * (0 - qvel)
    #exceed_limit_mask = jnp.where(qpos < ids["jnt_limits"][0, :] * 0.99, 1.0, 0.0) + jnp.where(qpos > ids["jnt_limits"][1, :] * 0.99, 1.0, 0.0)
    pd_tau = p_tau + d_tau
    #pd_tau = pd_tau

    #u_final = u * (pd_weight) + pd_tau * (1.0 - pd_weight)
    #u = jnp.clip(u, -ids["tau_limits"], ids["tau_limits"])

    return u_ff, pd_tau, des_pos, f, nle_ff