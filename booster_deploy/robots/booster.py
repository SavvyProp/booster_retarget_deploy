from ..controllers.controller_cfg import PrepareStateCfg, RobotCfg

T1_29DOF_CFG = RobotCfg(
    name="Booster_T1_29DOF",
    joint_names=
        [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll",
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
        "AL4",
        "AL5",
        "AL6",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "AR4",
        "AR5",
        "AR6",
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
        19.7392,  # AAHead_yaw
        19.7392,  # Head_pitch
        19.7392,  # Left_Shoulder_Pitch
        19.7392,  # Left_Shoulder_Roll
        19.7392,  # Left_Elbow_Pitch
        19.7392,  # Left_Elbow_Yaw
        19.7392,  # Left_Wrist_Pitch
        19.7392,  # Left_Wrist_Yaw
        19.7392,  # Left_Hand_Roll
        19.7392,  # Right_Shoulder_Pitch
        19.7392,  # Right_Shoulder_Roll
        19.7392,  # Right_Elbow_Pitch
        19.7392,  # Right_Elbow_Yaw
        19.7392,  # Right_Wrist_Pitch
        19.7392,  # Right_Wrist_Yaw
        19.7392,  # Right_Hand_Roll
        39.4784,  # Waist
        98.6960,  # Left_Hip_Pitch
        39.4784,  # Left_Hip_Roll
        39.4784,  # Left_Hip_Yaw
        98.6960,  # Left_Knee_Pitch
        19.7392,  # Left_Ankle_Pitch
        19.7392,  # Left_Ankle_Roll
        98.6960,  # Right_Hip_Pitch
        39.4784,  # Right_Hip_Roll
        39.4784,  # Right_Hip_Yaw
        98.6960,  # Right_Knee_Pitch
        19.7392,  # Right_Ankle_Pitch
        19.7392,  # Right_Ankle_Roll
    ],
    joint_damping=[
        1.2566,  # AAHead_yaw
        1.2566,  # Head_pitch
        1.2566,  # Left_Shoulder_Pitch
        1.2566,  # Left_Shoulder_Roll
        1.2566,  # Left_Elbow_Pitch
        1.2566,  # Left_Elbow_Yaw
        1.2566,  # Left_Wrist_Pitch
        1.2566,  # Left_Wrist_Yaw
        1.2566,  # Left_Hand_Roll
        1.2566,  # Right_Shoulder_Pitch
        1.2566,  # Right_Shoulder_Roll
        1.2566,  # Right_Elbow_Pitch
        1.2566,  # Right_Elbow_Yaw
        1.2566,  # Right_Wrist_Pitch
        1.2566,  # Right_Wrist_Yaw
        1.2566,  # Right_Hand_Roll
        2.5133,  # Waist
        6.2832,  # Left_Hip_Pitch
        2.5133,  # Left_Hip_Roll
        2.5133,  # Left_Hip_Yaw
        6.2832,  # Left_Knee_Pitch
        4.0,  # Left_Ankle_Pitch
        4.0,  # Left_Ankle_Roll
        6.2832,  # Right_Hip_Pitch
        2.5133,  # Right_Hip_Roll
        2.5133,  # Right_Hip_Yaw
        6.2832,  # Right_Knee_Pitch
        4.0,  # Right_Ankle_Pitch
        4.0,  # Right_Ankle_Roll
    ],
    effort_limit=[
        7,   # AAHead_yaw
        7,   # Head_pitch
        18,  # Left_Shoulder_Pitch
        18,  # Left_Shoulder_Roll
        18,  # Left_Elbow_Pitch
        18,  # Left_Elbow_Yaw
        18,  # Left_Wrist_Pitch
        18,  # Left_Wrist_Yaw
        18,  # Left_Hand_Roll
        18,  # Right_Shoulder_Pitch
        18,  # Right_Shoulder_Roll
        18,  # Right_Elbow_Pitch
        18,  # Right_Elbow_Yaw
        18,  # Right_Wrist_Pitch
        18,  # Right_Wrist_Yaw
        18,  # Right_Hand_Roll
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
        0.0, 0.0, 0.0,
        0.2, 1.35, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.0,
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
          'Left_Wrist_Pitch', 'Right_Wrist_Pitch', 
          'Left_Knee_Pitch', 'Right_Knee_Pitch', 
          'Left_Wrist_Yaw', 'Right_Wrist_Yaw', 
          'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 
          'Left_Hand_Roll', 'Right_Hand_Roll', 
          'Left_Ankle_Roll', 'Right_Ankle_Roll'],
    sim_body_names=[],
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_29dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20., 20., 20., 20.,
            40., 50., 20., 20., 20., 20., 20., 100.,
            350., 350., 180., 350., 250., 250.,
            350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            0.65, 0.65,
            0.5, 1.5, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2, 0.2, 0.2, 0.2,
            5.,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],

        joint_pos=[
        0.0, 0.0,
        0.2, -1.35, 0.0, -0.5,
        0.0, 0.0, 0.0,
        0.2, 1.35, 0.0,
        0.5, 0.0, 0.0,
        0.0, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    ]
    ),
)

GAIN_FAC = 0.75

T1_29DOF_LCC_CFG = RobotCfg(
    name="Booster_T1_29DOF",
    joint_names=
        [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Wrist_Pitch",
    "Right_Wrist_Yaw",
    "Right_Hand_Roll",
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
        "AL4",
        "AL5",
        "AL6",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "AR4",
        "AR5",
        "AR6",
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
        19.7392 * GAIN_FAC,  # AAHead_yaw
        19.7392 * GAIN_FAC,  # Head_pitch
        19.7392 * GAIN_FAC,  # Left_Shoulder_Pitch
        19.7392 * GAIN_FAC,  # Left_Shoulder_Roll
        19.7392 * GAIN_FAC,  # Left_Elbow_Pitch
        19.7392 * GAIN_FAC,  # Left_Elbow_Yaw
        19.7392 * GAIN_FAC,  # Left_Wrist_Pitch
        19.7392 * GAIN_FAC,  # Left_Wrist_Yaw
        19.7392 * GAIN_FAC,  # Left_Hand_Roll
        19.7392 * GAIN_FAC,  # Right_Shoulder_Pitch
        19.7392 * GAIN_FAC,  # Right_Shoulder_Roll
        19.7392 * GAIN_FAC,  # Right_Elbow_Pitch
        19.7392 * GAIN_FAC,  # Right_Elbow_Yaw
        19.7392 * GAIN_FAC,  # Right_Wrist_Pitch
        19.7392 * GAIN_FAC,  # Right_Wrist_Yaw
        19.7392 * GAIN_FAC,  # Right_Hand_Roll
        39.4784 * GAIN_FAC,  # Waist
        98.6960 * GAIN_FAC,  # Left_Hip_Pitch
        39.4784 * GAIN_FAC,  # Left_Hip_Roll
        39.4784 * GAIN_FAC,  # Left_Hip_Yaw
        98.6960 * GAIN_FAC,  # Left_Knee_Pitch
        19.7392 * GAIN_FAC,  # Left_Ankle_Pitch
        19.7392 * GAIN_FAC,  # Left_Ankle_Roll
        98.6960 * GAIN_FAC,  # Right_Hip_Pitch
        39.4784 * GAIN_FAC,  # Right_Hip_Roll
        39.4784 * GAIN_FAC,  # Right_Hip_Yaw
        98.6960 * GAIN_FAC,  # Right_Knee_Pitch
        19.7392 * GAIN_FAC,  # Right_Ankle_Pitch
        19.7392 * GAIN_FAC,  # Right_Ankle_Roll
    ],
    joint_damping=[
        1.2566,  # AAHead_yaw
        1.2566,  # Head_pitch
        1.2566,  # Left_Shoulder_Pitch
        1.2566,  # Left_Shoulder_Roll
        1.2566,  # Left_Elbow_Pitch
        1.2566,  # Left_Elbow_Yaw
        1.2566,  # Left_Wrist_Pitch
        1.2566,  # Left_Wrist_Yaw
        1.2566,  # Left_Hand_Roll
        1.2566,  # Right_Shoulder_Pitch
        1.2566,  # Right_Shoulder_Roll
        1.2566,  # Right_Elbow_Pitch
        1.2566,  # Right_Elbow_Yaw
        1.2566,  # Right_Wrist_Pitch
        1.2566,  # Right_Wrist_Yaw
        1.2566,  # Right_Hand_Roll
        2.5133,  # Waist
        6.2832,  # Left_Hip_Pitch
        2.5133,  # Left_Hip_Roll
        2.5133,  # Left_Hip_Yaw
        6.2832,  # Left_Knee_Pitch
        4.0,  # Left_Ankle_Pitch
        4.0,  # Left_Ankle_Roll
        6.2832,  # Right_Hip_Pitch
        2.5133,  # Right_Hip_Roll
        2.5133,  # Right_Hip_Yaw
        6.2832,  # Right_Knee_Pitch
        4.0,  # Right_Ankle_Pitch
        4.0,  # Right_Ankle_Roll
    ],
    effort_limit=[
        7,   # AAHead_yaw
        7,   # Head_pitch
        18,  # Left_Shoulder_Pitch
        18,  # Left_Shoulder_Roll
        18,  # Left_Elbow_Pitch
        18,  # Left_Elbow_Yaw
        18,  # Left_Wrist_Pitch
        18,  # Left_Wrist_Yaw
        18,  # Left_Hand_Roll
        18,  # Right_Shoulder_Pitch
        18,  # Right_Shoulder_Roll
        18,  # Right_Elbow_Pitch
        18,  # Right_Elbow_Yaw
        18,  # Right_Wrist_Pitch
        18,  # Right_Wrist_Yaw
        18,  # Right_Hand_Roll
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
        0.0, 0.0, 0.0,
        0.2, 1.35, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.0,
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
          'Left_Wrist_Pitch', 'Right_Wrist_Pitch', 
          'Left_Knee_Pitch', 'Right_Knee_Pitch', 
          'Left_Wrist_Yaw', 'Right_Wrist_Yaw', 
          'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 
          'Left_Hand_Roll', 'Right_Hand_Roll', 
          'Left_Ankle_Roll', 'Right_Ankle_Roll'],
    sim_body_names=[],
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_29dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20., 20., 20., 20.,
            40., 50., 20., 20., 20., 20., 20., 100.,
            350., 350., 180., 350., 250., 250.,
            350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            0.65, 0.65,
            0.5, 1.5, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2, 0.2, 0.2, 0.2,
            5.,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],

        joint_pos=[
        0.0, 0.0,
        0.2, -1.35, 0.0, -0.5,
        0.0, 0.0, 0.0,
        0.2, 1.35, 0.0,
        0.5, 0.0, 0.0,
        0.0, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    ]
    ),
)