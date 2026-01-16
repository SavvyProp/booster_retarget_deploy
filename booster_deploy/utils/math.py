import numpy as np


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