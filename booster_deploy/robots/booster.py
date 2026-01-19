from ..controllers.controller_cfg import PrepareStateCfg, RobotCfg


K1_CFG = RobotCfg(
    name="Booster_K1",
    joint_names=[
        "AAHead_yaw",
        "Head_pitch",
        "ALeft_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "ARight_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
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
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    ],
    joint_stiffness=[
        4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        80., 80.0, 80., 80., 30., 30.,
        80., 80.0, 80., 80., 30., 30.,
    ],
    joint_damping=[
        1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2.,
    ],
    default_joint_pos=[
        0, 0,
        0.0, -1.3, 0, -0.,
        0.0, 1.3, 0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.
    ],
    effort_limit=[
        6, 6,
        14, 14, 14, 14,
        14, 14, 14, 14,
        30, 35, 20, 40, 20, 20,
        30, 35, 20, 40, 20, 20,
    ],
    sim_joint_names=[       # joint order in isaacsim/isaaclab
        "AAHead_yaw",
        "ALeft_Shoulder_Pitch",
        "ARight_Shoulder_Pitch",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Head_pitch",
        "Left_Shoulder_Roll",
        "Right_Shoulder_Roll",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Elbow_Yaw",
        "Right_Elbow_Yaw",
        "Left_Knee_Pitch",
        "Right_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Right_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
    ],
    sim_body_names=[    # body order in isaacsim/isaaclab
        "Trunk",
        "Head_1",
        "Left_Arm_1",
        "Right_Arm_1",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Head_2",
        "Left_Arm_2",
        "Right_Arm_2",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Arm_3",
        "Right_Arm_3",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "left_hand_link",
        "right_hand_link",
        "Left_Shank",
        "Right_Shank",
        "Left_Ankle_Cross",
        "Right_Ankle_Cross",
        "left_foot_link",
        "right_foot_link",
    ],
    # {BOOSTER_ASSETS_DIR} will be replaced with
    # booster_assets.BOOSTER_ASSETS_DIR by MujocoController
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20,
            40., 50., 20., 20,
            350., 350., 180., 350., 250., 250.,
            350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            1.5, 1.5,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],
        joint_pos=[
            0, 0,
            0.0, -1.3, 0, -0.,
            0.0, 1.3, 0, 0.,
            -0.0, 0, 0, 0.105, -0.10, 0.,
            -0.0, 0, 0, 0.105, -0.10, 0.
        ],
    ),
)


T1_23DOF_CFG = RobotCfg(
    name="Booster_T1_23DOF",
    joint_names=[
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
        4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        80.,
        80., 80.0, 80., 80., 30., 30.,
        80., 80.0, 80., 80., 30., 30.,
    ],
    joint_damping=[
        1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        2.,
        2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2.,
    ],
    default_joint_pos=[
        0, 0,
        0.0, -1.4, 0, -0.,
        0.0, 1.4, 0, 0.,
        0.,
        -0.0, 0, 0, 0.0, -0.0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.
    ],
    effort_limit=[
        7, 7,
        18, 18, 18, 18,
        18, 18, 18, 18,
        25.,
        45, 25, 25, 60, 24, 15,
        45, 25, 25, 60, 24, 15,
    ],
    sim_joint_names=[       # joint order in isaacsim/isaaclab
        "AAHead_yaw",
        'Left_Shoulder_Pitch',
        'Right_Shoulder_Pitch',
        'Waist',
        "Head_pitch",
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
        'Left_Knee_Pitch',
        'Right_Knee_Pitch',
        'Left_Ankle_Pitch',
        'Right_Ankle_Pitch',
        'Left_Ankle_Roll',
        'Right_Ankle_Roll'
    ],

    sim_body_names=[    # body order in isaacsim/isaaclab

    ],
    # {BOOSTER_ASSETS_DIR} will be replaced with
    # booster_assets.BOOSTER_ASSETS_DIR by MujocoController
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20,
            40., 50., 20., 20,
            350.,
            350., 350., 180., 350., 350., 350.,
            350., 350., 180., 350., 350., 350.,],
        damping=[
            0.65, 0.65,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            5.,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],
        joint_pos=[
            0, 0,
            0.0, -1.4, 0, -0.,
            0.0, 1.4, 0, 0.,
            0.,
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0
        ],
    ),
)

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
        1.2566,  # Left_Ankle_Pitch
        1.2566,  # Left_Ankle_Roll
        6.2832,  # Right_Hip_Pitch
        2.5133,  # Right_Hip_Roll
        2.5133,  # Right_Hip_Yaw
        6.2832,  # Right_Knee_Pitch
        1.2566,  # Right_Ankle_Pitch
        1.2566,  # Right_Ankle_Roll
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