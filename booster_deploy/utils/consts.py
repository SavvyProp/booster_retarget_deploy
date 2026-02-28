from dataclasses import dataclass
import re

import numpy as np

NP_FLOAT = np.float32


@dataclass(frozen=True)
class BoosterConsts:
    t1_lg_action_scale: dict[str, float]
    t1_action_scale: dict[str, float]
    joint_names: list[str]
    isaac_joint_names: list[str]
    isaac_ankle_joint_names: list[str]
    action_scale_list: list[float]
    lg_action_scale_list: list[float]
    action_scale: np.ndarray
    lg_action_scale: np.ndarray
    name_to_mj_idx: dict[str, int]
    name_to_isaac_idx: dict[str, int]
    name_to_isaac_ankle_idx: dict[str, int]
    isaac_to_mj: np.ndarray
    mj_to_isaac: np.ndarray
    mj_to_isaac_ankle: np.ndarray
    joint_impedance_by_name: dict[str, tuple[float, float, float]]
    joint_torques: np.ndarray
    is_joint_pos: np.ndarray
    is_joint_pos_ankle: np.ndarray
    angular_inertia: np.ndarray
    mass: float


def _build_scale_list(jnames: list[str], scale_by_pattern: dict[str, float]) -> list[float]:
    scale_list: list[float] = []
    for jname in jnames:
        scale = 1.0
        for pattern, val in scale_by_pattern.items():
            if re.fullmatch(pattern, jname):
                scale = val
                break
        scale_list.append(scale)
    return scale_list


def _build_booster_consts() -> BoosterConsts:
    t1_lg_action_scale = {'.*_Hip_Pitch': 0.12665147956016215, '.*_Hip_Roll': 0.10554289963346844, '.*_Hip_Yaw': 0.10554289963346844, '.*_Knee_Pitch': 0.16886863941354954, 'Waist': 0.12665147956016212, '.*_Ankle_Pitch': 0.2026423672962594, '.*_Ankle_Roll': 0.12665147956016212, '.*_Shoulder_Pitch': 0.15198177547219455, '.*_Shoulder_Roll': 0.15198177547219455, '.*_Elbow_Pitch': 0.15198177547219455, '.*_Elbow_Yaw': 0.15198177547219455, 'AAHead_yaw': 0.08443431970677476, 'Head_pitch': 0.08443431970677476}
    t1_action_scale = {'.*_Hip_Pitch': 0.09498860967012161, '.*_Hip_Roll': 0.0949886096701216, '.*_Hip_Yaw': 0.0949886096701216, '.*_Knee_Pitch': 0.12665147956016215, 'Waist': 0.0949886096701216, '.*_Ankle_Pitch': 0.15198177547219457, '.*_Ankle_Roll': 0.0949886096701216, '.*_Shoulder_Pitch': 0.11398633160414592, '.*_Shoulder_Roll': 0.11398633160414592, '.*_Elbow_Pitch': 0.11398633160414592, '.*_Elbow_Yaw': 0.11398633160414592, 'AAHead_yaw': 0.06332573978008108, 'Head_pitch': 0.06332573978008108}

    joint_names = [
        "AAHead_yaw",
        "Head_pitch",
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ]

    isaac_joint_names = [
        "AAHead_yaw",
        "Left_Shoulder_Pitch",
        "Right_Shoulder_Pitch",
        "Waist",
        "Head_pitch",
        "Left_Shoulder_Roll",
        "Right_Shoulder_Roll",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Elbow_Yaw",
        "Right_Elbow_Yaw",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Knee_Pitch",
        "Right_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Right_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
    ]

    isaac_ankle_joint_names = [
        "Left_Ankle_Pitch",
        "Right_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
        "AAHead_yaw",
        "Left_Shoulder_Pitch",
        "Right_Shoulder_Pitch",
        "Waist",
        "Head_pitch",
        "Left_Shoulder_Roll",
        "Right_Shoulder_Roll",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Elbow_Yaw",
        "Right_Elbow_Yaw",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Knee_Pitch",
        "Right_Knee_Pitch",
    ]

    action_scale_list = _build_scale_list(joint_names, t1_action_scale)
    lg_action_scale_list = _build_scale_list(joint_names, t1_lg_action_scale)
    action_scale = np.array(action_scale_list, dtype=NP_FLOAT)
    lg_action_scale = np.array(lg_action_scale_list, dtype=NP_FLOAT)

    name_to_mj_idx = {name: i for i, name in enumerate(joint_names)}
    name_to_isaac_idx = {name: i for i, name in enumerate(isaac_joint_names)}
    name_to_isaac_ankle_idx = {name: i for i, name in enumerate(isaac_ankle_joint_names)}

    isaac_to_mj = np.array(
        [name_to_isaac_idx[name] for name in joint_names],
        dtype=int,
    )
    mj_to_isaac = np.array(
        [name_to_mj_idx[name] for name in isaac_joint_names],
        dtype=int,
    )
    mj_to_isaac_ankle = np.array(
        [name_to_mj_idx[name] for name in isaac_ankle_joint_names],
        dtype=int,
    )

    # [kp, kd, torque_limit] by joint name
    joint_impedance_by_name = {
        "AAHead_yaw": (19.7392, 1.2566, 7.0),
        "Head_pitch": (19.7392, 1.2566, 7.0),
        "Left_Shoulder_Pitch": (19.7392, 1.2566, 18.0),
        "Left_Shoulder_Roll": (19.7392, 1.2566, 18.0),
        "Left_Elbow_Pitch": (19.7392, 1.2566, 18.0),
        "Left_Elbow_Yaw": (19.7392, 1.2566, 18.0),
        "Left_Wrist_Pitch": (19.7392, 1.2566, 18.0),
        "Left_Wrist_Yaw": (19.7392, 1.2566, 18.0),
        "Left_Hand_Roll": (19.7392, 1.2566, 18.0),
        "Right_Shoulder_Pitch": (19.7392, 1.2566, 18.0),
        "Right_Shoulder_Roll": (19.7392, 1.2566, 18.0),
        "Right_Elbow_Pitch": (19.7392, 1.2566, 18.0),
        "Right_Elbow_Yaw": (19.7392, 1.2566, 18.0),
        "Right_Wrist_Pitch": (19.7392, 1.2566, 18.0),
        "Right_Wrist_Yaw": (19.7392, 1.2566, 18.0),
        "Right_Hand_Roll": (19.7392, 1.2566, 18.0),
        "Waist": (39.4784, 2.5133, 30.0),
        "Left_Hip_Pitch": (98.6960, 6.2832, 45.0),
        "Left_Hip_Roll": (39.4784, 2.5133, 25.0),
        "Left_Hip_Yaw": (39.4784, 2.5133, 25.0),
        "Left_Knee_Pitch": (98.6960, 6.2832, 60.0),
        "Left_Ankle_Pitch": (19.7392, 1.2566, 24.0),
        "Left_Ankle_Roll": (19.7392, 1.2566, 15.0),
        "Right_Hip_Pitch": (98.6960, 6.2832, 45.0),
        "Right_Hip_Roll": (39.4784, 2.5133, 25.0),
        "Right_Hip_Yaw": (39.4784, 2.5133, 25.0),
        "Right_Knee_Pitch": (98.6960, 6.2832, 60.0),
        "Right_Ankle_Pitch": (19.7392, 1.2566, 24.0),
        "Right_Ankle_Roll": (19.7392, 1.2566, 15.0),
    }
    missing_impedance = [name for name in joint_names if name not in joint_impedance_by_name]
    if missing_impedance:
        raise KeyError(f"Missing impedance data for joints: {missing_impedance}")
    joint_torques = np.array(
        [joint_impedance_by_name[name][2] for name in joint_names],
        dtype=NP_FLOAT,
    )

    is_joint_pos = np.array(
        [
            0.0,
            0.2,
            0.2,
            0.0,
            0.0,
            -1.35,
            1.35,
            -0.2,
            -0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.5,
            0.5,
            0.0,
            0.0,
            0.42,
            0.42,
            -0.23,
            -0.23,
            0.0,
            0.0,
        ],
        dtype=NP_FLOAT,
    )
    is_joint_pos_ankle = np.array(
        [
            -0.23,
            -0.23,
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.0,
            0.0,
            -1.35,
            1.35,
            -0.2,
            -0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.5,
            0.5,
            0.0,
            0.0,
            0.42,
            0.42,
        ],
        dtype=NP_FLOAT,
    )
    angular_inertia = np.array(
        [
            [2.76900149e00, 4.50170509e-04, 3.66299529e-02],
            [4.50170509e-04, 2.30203655e00, -4.42839862e-04],
            [3.66299529e-02, -4.42839862e-04, 5.62235551e-01],
        ],
        dtype=NP_FLOAT,
    )
    mass = 31.614357

    return BoosterConsts(
        t1_lg_action_scale=t1_lg_action_scale,
        t1_action_scale=t1_action_scale,
        joint_names=joint_names,
        isaac_joint_names=isaac_joint_names,
        isaac_ankle_joint_names=isaac_ankle_joint_names,
        action_scale_list=action_scale_list,
        lg_action_scale_list=lg_action_scale_list,
        action_scale=action_scale,
        lg_action_scale=lg_action_scale,
        name_to_mj_idx=name_to_mj_idx,
        name_to_isaac_idx=name_to_isaac_idx,
        name_to_isaac_ankle_idx=name_to_isaac_ankle_idx,
        isaac_to_mj=isaac_to_mj,
        mj_to_isaac=mj_to_isaac,
        mj_to_isaac_ankle=mj_to_isaac_ankle,
        joint_impedance_by_name=joint_impedance_by_name,
        joint_torques=joint_torques,
        is_joint_pos=is_joint_pos,
        is_joint_pos_ankle=is_joint_pos_ankle,
        angular_inertia=angular_inertia,
        mass=mass,
    )


BOOSTER_CONSTS = _build_booster_consts()

# Backward-compatible module-level aliases.
T1_LG_ACTION_SCALE = BOOSTER_CONSTS.t1_lg_action_scale
T1_ACTION_SCALE = BOOSTER_CONSTS.t1_action_scale
joint_names = BOOSTER_CONSTS.joint_names
isaac_joint_names = BOOSTER_CONSTS.isaac_joint_names
isaac_ankle_joint_names = BOOSTER_CONSTS.isaac_ankle_joint_names
action_scale_list = BOOSTER_CONSTS.action_scale_list
lg_action_scale_list = BOOSTER_CONSTS.lg_action_scale_list
action_scale = BOOSTER_CONSTS.action_scale
lg_action_scale = BOOSTER_CONSTS.lg_action_scale
name_to_mj_idx = BOOSTER_CONSTS.name_to_mj_idx
name_to_isaac_idx = BOOSTER_CONSTS.name_to_isaac_idx
name_to_isaac_ankle_idx = BOOSTER_CONSTS.name_to_isaac_ankle_idx
isaac_to_mj = BOOSTER_CONSTS.isaac_to_mj
mj_to_isaac = BOOSTER_CONSTS.mj_to_isaac
mj_to_isaac_ankle = BOOSTER_CONSTS.mj_to_isaac_ankle
JOINT_IMPEDANCE_BY_NAME = BOOSTER_CONSTS.joint_impedance_by_name
joint_torques = BOOSTER_CONSTS.joint_torques
is_joint_pos = BOOSTER_CONSTS.is_joint_pos
is_joint_pos_ankle = BOOSTER_CONSTS.is_joint_pos_ankle
angular_inertia = BOOSTER_CONSTS.angular_inertia
mass = BOOSTER_CONSTS.mass
