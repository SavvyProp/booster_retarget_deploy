import numpy as np

def rotate_vector_rpy(roll, pitch, yaw, vector):
    """
    Rotate a vector by the given roll, pitch, and yaw angles.

    Parameters:
    roll (float): The roll angle in radians.
    pitch (float): The pitch angle in radians.
    yaw (float): The yaw angle in radians.
    vector (np.ndarray): The 3D vector to be rotated.

    Returns:
    np.ndarray: The rotated 3D vector.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R_z @ R_y @ R_x) @ vector

def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray, *, ordering: str = "wxyz") -> np.ndarray:
    """Rotate a 3D vector by a quaternion.

    Args:
        v: 3D vector, shape (3,).
        q: Quaternion, shape (4,). Default ordering is MuJoCo-style (w, x, y, z).
        ordering: "wxyz" (default) or "xyzw".

    Returns:
        Rotated 3D vector, shape (3,).

    Notes:
        Uses the efficient formula: v' = v + 2*cross(q_vec, cross(q_vec, v) + w*v)
        Assumes q is a unit quaternion; it will be normalized for safety.
    """
    v = np.asarray(v, dtype=np.float64).reshape(3)
    q = np.asarray(q, dtype=np.float64).reshape(4)

    if ordering == "xyzw":
        x, y, z, w = q
    elif ordering == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError(f"Unknown quaternion ordering '{ordering}'. Use 'wxyz' or 'xyzw'.")

    # Normalize quaternion (robustness)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        raise ValueError("Quaternion has zero norm.")
    w, x, y, z = w / n, x / n, y / n, z / n

    q_vec = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(q_vec, v)
    v_rot = v + w * t + np.cross(q_vec, t)
    return v_rot.astype(v.dtype, copy=False)


def quat_mul(q1: np.ndarray, q2: np.ndarray, *, ordering: str = "wxyz", normalize: bool = False) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        q1: Quaternion, shape (4,).
        q2: Quaternion, shape (4,).
        ordering: "wxyz" (default) or "xyzw".
        normalize: If True, normalize the output quaternion.

    Returns:
        Product quaternion q = q1 * q2 in the same ordering, shape (4,).

    Notes:
        This uses Hamilton product. If quaternions represent rotations, the product
        corresponds to composition (apply q2 then q1) under the usual convention.
    """
    q1 = np.asarray(q1, dtype=np.float32).reshape(4)
    q2 = np.asarray(q2, dtype=np.float32).reshape(4)

    if ordering == "xyzw":
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
    elif ordering == "wxyz":
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
    else:
        raise ValueError(f"Unknown quaternion ordering '{ordering}'. Use 'wxyz' or 'xyzw'.")

    # Hamilton product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    if normalize:
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n <= 0.0:
            raise ValueError("Quaternion product has zero norm.")
        w, x, y, z = w / n, x / n, y / n, z / n

    if ordering == "xyzw":
        return np.array([x, y, z, w], dtype=np.float64)
    return np.array([w, x, y, z], dtype=np.float64)


def rotmat_to_quat(R: np.ndarray, *, ordering: str = "wxyz", normalize: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.

    Args:
        R: Rotation matrix, shape (3, 3).
        ordering: "wxyz" (default, MuJoCo-style) or "xyzw".
        normalize: If True, normalize the output quaternion.
        eps: Small threshold for numerical stability.

    Returns:
        Quaternion representing the same rotation, shape (4,), in requested ordering.

    Notes:
        Uses a branch on the matrix trace (and diagonal dominance) for numerical stability.
        The returned quaternion has an arbitrary sign (q and -q represent the same rotation).
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R.shape}.")

    trace = float(np.trace(R))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0  # s = 4*w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Find the largest diagonal element and proceed accordingly
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(max(eps, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0  # s = 4*x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(max(eps, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0  # s = 4*y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(max(eps, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0  # s = 4*z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q_wxyz = np.array([w, x, y, z], dtype=np.float64)

    if normalize:
        n = np.linalg.norm(q_wxyz)
        if n <= 0.0:
            raise ValueError("Rotation matrix converted to a zero-norm quaternion.")
        q_wxyz = q_wxyz / n

    ordering = ordering.lower()
    if ordering == "wxyz":
        return q_wxyz
    if ordering == "xyzw":
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    raise ValueError(f"Unknown quaternion ordering '{ordering}'. Use 'wxyz' or 'xyzw'.")


def rotmat_to_rpy(R: np.ndarray, *, order: str = "zyx", eps: float = 1e-8) -> np.ndarray:
    """Convert a rotation matrix to roll/pitch/yaw (radians).

    Args:
        R: Rotation matrix, shape (3, 3).
        order: Axis order / convention. "zyx" (default) returns (roll, pitch, yaw)
            for the intrinsic Z-Y-X (yaw-pitch-roll) sequence, commonly used in robotics.
            "xyz" returns (roll, pitch, yaw) for intrinsic X-Y-Z.
        eps: Small threshold to detect gimbal lock.

    Returns:
        np.ndarray of shape (3,) with (roll, pitch, yaw) in radians.

    Notes:
        - For order="zyx": R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        - For order="xyz": R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
        - In gimbal-lock configurations, yaw (or roll) is set to 0 and the remaining
          angle is recovered from available matrix terms.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R.shape}.")

    order = order.lower()

    if order == "zyx":
        # pitch = asin(-R[2,0])
        s = -R[2, 0]
        s = np.clip(s, -1.0, 1.0)
        pitch = np.arcsin(s)
        cp = np.cos(pitch)

        if abs(cp) > eps:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock: cp ~ 0. Choose yaw = 0 and solve roll from remaining terms.
            yaw = 0.0
            # When pitch ~ +pi/2: R[0,1] = -sin(roll), R[0,2] = cos(roll)
            # When pitch ~ -pi/2: R[0,1] =  sin(roll), R[0,2] = cos(roll)
            roll = np.arctan2(-R[0, 1], R[0, 2])

        return np.array([roll, pitch, yaw], dtype=np.float64)

    if order == "xyz":
        # For R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
        # pitch = asin(R[0,2])
        s = R[0, 2]
        s = np.clip(s, -1.0, 1.0)
        pitch = np.arcsin(s)
        cp = np.cos(pitch)

        if abs(cp) > eps:
            roll = np.arctan2(-R[1, 2], R[2, 2])
            yaw = np.arctan2(-R[0, 1], R[0, 0])
        else:
            # Gimbal lock: cp ~ 0. Choose yaw = 0 and solve roll.
            yaw = 0.0
            roll = np.arctan2(R[2, 1], R[1, 1])

        return np.array([roll, pitch, yaw], dtype=np.float64)

    raise ValueError(f"Unknown order '{order}'. Use 'zyx' or 'xyz'.")


def quat_conjugate(q: np.ndarray, *, ordering: str = "wxyz") -> np.ndarray:
    """Return the quaternion conjugate.

    For unit quaternions, the conjugate is the inverse.

    Args:
        q: Quaternion, shape (4,).
        ordering: "wxyz" (default) or "xyzw".

    Returns:
        Conjugated quaternion in the same ordering.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    ordering = ordering.lower()

    if ordering == "wxyz":
        w, x, y, z = q
        return np.array([w, -x, -y, -z], dtype=np.float64)
    if ordering == "xyzw":
        x, y, z, w = q
        return np.array([-x, -y, -z, w], dtype=np.float64)

    raise ValueError(f"Unknown quaternion ordering '{ordering}'. Use 'wxyz' or 'xyzw'.")


def quat_inverse(q: np.ndarray, *, ordering: str = "wxyz") -> np.ndarray:
    """Return the quaternion inverse.

    Args:
        q: Quaternion, shape (4,).
        ordering: "wxyz" (default) or "xyzw".

    Returns:
        Inverted quaternion in the same ordering.

    Notes:
        For unit quaternions, inverse equals conjugate.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if ordering.lower() == "wxyz":
        w, x, y, z = q
        n2 = w * w + x * x + y * y + z * z
        if n2 <= 0.0:
            raise ValueError("Quaternion has zero norm.")
        return np.array([w, -x, -y, -z], dtype=np.float64) / n2

    if ordering.lower() == "xyzw":
        x, y, z, w = q
        n2 = w * w + x * x + y * y + z * z
        if n2 <= 0.0:
            raise ValueError("Quaternion has zero norm.")
        return np.array([-x, -y, -z, w], dtype=np.float64) / n2

    raise ValueError(f"Unknown quaternion ordering '{ordering}'. Use 'wxyz' or 'xyzw'.")


def world_to_base_vector(v_world: np.ndarray, base_quat_wb: np.ndarray, *, ordering: str = "wxyz") -> np.ndarray:
    """Convert a 3D vector from world frame to base frame using the base quaternion.

    Args:
        v_world: Vector expressed in world frame, shape (3,).
        base_quat_wb: Base orientation quaternion representing the rotation from base->world
            (i.e., orientation of the base frame expressed in world coordinates), shape (4,).
            This matches MuJoCo qpos quaternion convention for a free joint.
        ordering: "wxyz" (default) or "xyzw".

    Returns:
        Vector expressed in the base frame, shape (3,).

    Notes:
        If q_wb rotates vectors from base to world (v_w = R(q_wb) v_b), then the world->base
        conversion is v_b = R(q_wb)^T v_w, which is equivalent to rotating by q_wb^{-1}.

        This function applies the inverse rotation explicitly via quaternion inverse.
    """
    q_bw = quat_inverse(base_quat_wb, ordering=ordering)
    return rotate_vector_by_quat(v_world, q_bw, ordering=ordering)