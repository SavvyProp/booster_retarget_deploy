from ..controllers.controller_cfg import PrepareStateCfg, RobotCfg

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

ARMATURE_ANK = 0.01
ARMATURE_HIGH = 0.03
ARMATURE_MID = 0.02
ARMATURE_LOW = 0.01

STIFFNESS_LOW = ARMATURE_LOW * NATURAL_FREQ**2
STIFFNESS_ANK = ARMATURE_ANK * NATURAL_FREQ**2
STIFFNESS_MID = ARMATURE_MID * NATURAL_FREQ**2
STIFFNESS_HIGH = ARMATURE_HIGH * NATURAL_FREQ**2

DAMPING_LOW = 2.0 * DAMPING_RATIO * ARMATURE_LOW * NATURAL_FREQ
DAMPING_ANK = 2.0 * DAMPING_RATIO * ARMATURE_ANK * NATURAL_FREQ
DAMPING_MID = 2.0 * DAMPING_RATIO * ARMATURE_MID * NATURAL_FREQ
DAMPING_HIGH = 2.0 * DAMPING_RATIO * ARMATURE_HIGH * NATURAL_FREQ


T1_23DOF_CFG = RobotCfg(
    name="Booster_T1_23DOF",
    joint_names=
        [
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
    ],
    body_names=[
        "Trunk",
        "H1",
        "H2",
        "AL1",
        "AL2",
        "AL3",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "right_hand_link",
        "Waist",
        "Hip_Pitch_Left",
        "Hip_Roll_Left",
        "Hip_Yaw_Left",
        "Shank_Left",
        "Ankle_Cross_Left",
        "left_foot_link",
        "Hip_Pitch_Right",
        "Hip_Roll_Right",
        "Hip_Yaw_Right",
        "Shank_Right",
        "Ankle_Cross_Right",
        "right_foot_link",
    ],
    joint_stiffness=[
        STIFFNESS_LOW,  # AAHead_yaw
        STIFFNESS_LOW,  # Head_pitch
        STIFFNESS_LOW,  # Left_Shoulder_Pitch
        STIFFNESS_LOW,  # Left_Shoulder_Roll
        STIFFNESS_LOW,  # Left_Elbow_Pitch
        STIFFNESS_LOW,  # Left_Elbow_Yaw
        STIFFNESS_LOW,  # Right_Shoulder_Pitch
        STIFFNESS_LOW,  # Right_Shoulder_Roll
        STIFFNESS_LOW,  # Right_Elbow_Pitch
        STIFFNESS_LOW,  # Right_Elbow_Yaw
        STIFFNESS_MID,  # Waist
        STIFFNESS_HIGH,  # Left_Hip_Pitch
        STIFFNESS_MID,  # Left_Hip_Roll
        STIFFNESS_MID,  # Left_Hip_Yaw
        STIFFNESS_HIGH,  # Left_Knee_Pitch
        STIFFNESS_ANK,  # Left_Ankle_Pitch
        STIFFNESS_ANK,  # Left_Ankle_Roll
        STIFFNESS_HIGH,  # Right_Hip_Pitch
        STIFFNESS_MID,  # Right_Hip_Roll
        STIFFNESS_MID,  # Right_Hip_Yaw
        STIFFNESS_HIGH,  # Right_Knee_Pitch
        STIFFNESS_ANK,  # Right_Ankle_Pitch
        STIFFNESS_ANK,  # Right_Ankle_Roll
    ],
    joint_damping=[
        DAMPING_LOW,  # AAHead_yaw
        DAMPING_LOW,  # Head_pitch
        DAMPING_LOW,  # Left_Shoulder_Pitch
        DAMPING_LOW,  # Left_Shoulder_Roll
        DAMPING_LOW,  # Left_Elbow_Pitch
        DAMPING_LOW,  # Left_Elbow_Yaw
        DAMPING_LOW,  # Right_Shoulder_Pitch
        DAMPING_LOW,  # Right_Shoulder_Roll
        DAMPING_LOW,  # Right_Elbow_Pitch
        DAMPING_LOW,  # Right_Elbow_Yaw
        DAMPING_MID,  # Waist
        DAMPING_HIGH,  # Left_Hip_Pitch
        DAMPING_MID,  # Left_Hip_Roll
        DAMPING_MID,  # Left_Hip_Yaw
        DAMPING_HIGH,  # Left_Knee_Pitch
        DAMPING_ANK,  # Left_Ankle_Pitch
        DAMPING_ANK,  # Left_Ankle_Roll
        DAMPING_HIGH,  # Right_Hip_Pitch
        DAMPING_MID,  # Right_Hip_Roll
        DAMPING_MID,  # Right_Hip_Yaw
        DAMPING_HIGH,  # Right_Knee_Pitch
        DAMPING_ANK,  # Right_Ankle_Pitch
        DAMPING_ANK,  # Right_Ankle_Roll
    ],
    effort_limit=[
        7,   # AAHead_yaw
        7,   # Head_pitch
        18,  # Left_Shoulder_Pitch
        18,  # Left_Shoulder_Roll
        18,  # Left_Elbow_Pitch
        18,  # Left_Elbow_Yaw
        18,  # Right_Shoulder_Pitch
        18,  # Right_Shoulder_Roll
        18,  # Right_Elbow_Pitch
        18,  # Right_Elbow_Yaw
        30,  # Waist
        45,  # Left_Hip_Pitch
        25,  # Left_Hip_Roll
        25,  # Left_Hip_Yaw
        60,  # Left_Knee_Pitch
        24,  # Left_Ankle_Pitch
        15,  # Left_Ankle_Roll
        45,  # Right_Hip_Pitch
        25,  # Right_Hip_Roll
        25,  # Right_Hip_Yaw
        60,  # Right_Knee_Pitch
        24,  # Right_Ankle_Pitch
        15,  # Right_Ankle_Roll
    ],
    default_joint_pos=[
        0.0, 0.0,
        0.2, -1.35, 0.0, -0.5,
        0.2, 1.35, 0.0, 0.5, 
        0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    ],

    sim_joint_names=['AAHead_yaw', 
          'Left_Shoulder_Pitch', 
          'Right_Shoulder_Pitch', 
          'Waist', 
          'Head_pitch', 
          'Left_Shoulder_Roll',
          'Right_Shoulder_Roll', 
          'Left_Hip_Pitch', 
          'Right_Hip_Pitch', 
          'Left_Elbow_Pitch', 
          'Right_Elbow_Pitch', 
          'Left_Hip_Roll', 
          'Right_Hip_Roll', 
          'Left_Elbow_Yaw', 
          'Right_Elbow_Yaw', 
          'Left_Hip_Yaw', 
          'Right_Hip_Yaw', 
          'Left_Knee_Pitch', 'Right_Knee_Pitch', 
          'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 
          'Left_Ankle_Roll', 'Right_Ankle_Roll'],
    sim_body_names=[],
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20.,
            40., 50., 20., 20., 100.,
            350., 350., 180., 350., 250., 250.,
            350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            0.65, 0.65,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            5.,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],

        joint_pos=[
        0.0, 0.0,
        0.2, -1.35, 0.0, -0.5,
        0.2, 1.35, 0.0, 0.5,
        0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    ]
    ),
)

GAIN_FAC = 0.85
DAMP_FAC = 1.0

T1_23DOF_LCC_CFG = RobotCfg(
    name="Booster_T1_29DOF",
    joint_names=
        [
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
    ],
    body_names=[
        "Trunk",
        "H1",
        "H2",
        "AL1",
        "AL2",
        "AL3",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "right_hand_link",
        "Waist",
        "Hip_Pitch_Left",
        "Hip_Roll_Left",
        "Hip_Yaw_Left",
        "Shank_Left",
        "Ankle_Cross_Left",
        "left_foot_link",
        "Hip_Pitch_Right",
        "Hip_Roll_Right",
        "Hip_Yaw_Right",
        "Shank_Right",
        "Ankle_Cross_Right",
        "right_foot_link",
    ],
    joint_stiffness=[
        STIFFNESS_LOW * GAIN_FAC,  # AAHead_yaw
        STIFFNESS_LOW * GAIN_FAC,  # Head_pitch
        STIFFNESS_LOW * GAIN_FAC,  # Left_Shoulder_Pitch
        STIFFNESS_LOW * GAIN_FAC,  # Left_Shoulder_Roll
        STIFFNESS_LOW * GAIN_FAC,  # Left_Elbow_Pitch
        STIFFNESS_LOW * GAIN_FAC,  # Left_Elbow_Yaw
        STIFFNESS_LOW * GAIN_FAC,  # Right_Shoulder_Pitch
        STIFFNESS_LOW * GAIN_FAC,  # Right_Shoulder_Roll
        STIFFNESS_LOW * GAIN_FAC,  # Right_Elbow_Pitch
        STIFFNESS_LOW * GAIN_FAC,  # Right_Elbow_Yaw
        STIFFNESS_MID * GAIN_FAC,  # Waist
        STIFFNESS_HIGH * GAIN_FAC,  # Left_Hip_Pitch
        STIFFNESS_MID * GAIN_FAC,  # Left_Hip_Roll
        STIFFNESS_MID * GAIN_FAC,  # Left_Hip_Yaw
        STIFFNESS_HIGH * GAIN_FAC,  # Left_Knee_Pitch
        STIFFNESS_ANK * GAIN_FAC,  # Left_Ankle_Pitch
        STIFFNESS_ANK * GAIN_FAC,  # Left_Ankle_Roll
        STIFFNESS_HIGH * GAIN_FAC,  # Right_Hip_Pitch
        STIFFNESS_MID * GAIN_FAC,  # Right_Hip_Roll
        STIFFNESS_MID * GAIN_FAC,  # Right_Hip_Yaw
        STIFFNESS_HIGH * GAIN_FAC,  # Right_Knee_Pitch
        STIFFNESS_ANK * GAIN_FAC,  # Right_Ankle_Pitch
        STIFFNESS_ANK * GAIN_FAC,  # Right_Ankle_Roll
    ],
    joint_damping=[
        DAMPING_LOW * DAMP_FAC,  # AAHead_yaw
        DAMPING_LOW * DAMP_FAC,  # Head_pitch
        DAMPING_LOW * DAMP_FAC,  # Left_Shoulder_Pitch
        DAMPING_LOW * DAMP_FAC,  # Left_Shoulder_Roll
        DAMPING_LOW * DAMP_FAC,  # Left_Elbow_Pitch
        DAMPING_LOW * DAMP_FAC,  # Left_Elbow_Yaw
        DAMPING_LOW * DAMP_FAC,  # Right_Shoulder_Pitch
        DAMPING_LOW * DAMP_FAC,  # Right_Shoulder_Roll
        DAMPING_LOW * DAMP_FAC,  # Right_Elbow_Pitch
        DAMPING_LOW * DAMP_FAC,  # Right_Elbow_Yaw
        DAMPING_MID * DAMP_FAC,  # Waist
        DAMPING_HIGH * DAMP_FAC,  # Left_Hip_Pitch
        DAMPING_MID * DAMP_FAC,  # Left_Hip_Roll
        DAMPING_MID * DAMP_FAC,  # Left_Hip_Yaw
        DAMPING_HIGH * DAMP_FAC,  # Left_Knee_Pitch
        DAMPING_ANK,  # Left_Ankle_Pitch
        DAMPING_ANK,  # Left_Ankle_Roll
        DAMPING_HIGH * DAMP_FAC,  # Right_Hip_Pitch
        DAMPING_MID * DAMP_FAC,  # Right_Hip_Roll
        DAMPING_MID * DAMP_FAC,  # Right_Hip_Yaw
        DAMPING_HIGH * DAMP_FAC,  # Right_Knee_Pitch
        DAMPING_ANK,  # Right_Ankle_Pitch
        DAMPING_ANK,  # Right_Ankle_Roll
    ],
    effort_limit=[
        7,   # AAHead_yaw
        7,   # Head_pitch
        18,  # Left_Shoulder_Pitch
        18,  # Left_Shoulder_Roll
        18,  # Left_Elbow_Pitch
        18,  # Left_Elbow_Yaw
        18,  # Right_Shoulder_Pitch
        18,  # Right_Shoulder_Roll
        18,  # Right_Elbow_Pitch
        18,  # Right_Elbow_Yaw
        30,  # Waist
        45,  # Left_Hip_Pitch
        25,  # Left_Hip_Roll
        25,  # Left_Hip_Yaw
        60,  # Left_Knee_Pitch
        24,  # Left_Ankle_Pitch
        15,  # Left_Ankle_Roll
        45,  # Right_Hip_Pitch
        25,  # Right_Hip_Roll
        25,  # Right_Hip_Yaw
        60,  # Right_Knee_Pitch
        24,  # Right_Ankle_Pitch
        15,  # Right_Ankle_Roll
    ],
    default_joint_pos=[
        0.0, 0.0,
        0.2, -1.35, 0.0, -0.5,
        0.2, 1.35, 0.0, 0.5, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    ],

    sim_joint_names=['AAHead_yaw', 
          'Left_Shoulder_Pitch', 
          'Right_Shoulder_Pitch', 
          'Waist', 
          'Head_pitch', 
          'Left_Shoulder_Roll',
          'Right_Shoulder_Roll', 
          'Left_Hip_Pitch', 
          'Right_Hip_Pitch', 
          'Left_Elbow_Pitch', 
          'Right_Elbow_Pitch', 
          'Left_Hip_Roll', 
          'Right_Hip_Roll', 
          'Left_Elbow_Yaw', 
          'Right_Elbow_Yaw', 
          'Left_Hip_Yaw', 
          'Right_Hip_Yaw',  
          'Left_Knee_Pitch', 'Right_Knee_Pitch', 
          'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 
          'Left_Ankle_Roll', 'Right_Ankle_Roll'],
    sim_body_names=[],
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20.,
            40., 50., 20., 20.,100.,
            350., 350., 180., 350., 250., 250.,
            350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            0.65, 0.65,
            0.5, 1.5, 0.2, 0.2, 
            0.5, 1.5, 0.2, 0.2, 
            5.,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],

        joint_pos=[
        0.0, 0.0,
        0.2, -1.35, 0.0, -0.5,
        0.2, 1.35, 0.0, 0.5, 
        0.0, 
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    ]
    ),
)
