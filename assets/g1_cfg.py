# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""G1 Humanoid Robot Asset Configuration.

This module provides pre-configured G1 robot articulation for Isaac Lab.
Adapted from Unitree's isaaclab implementation.

References:
    https://github.com/unitreerobotics/unitree_sim_isaaclab
"""

from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# G1 Robot Configuration
##

G1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    # TODO: Update with actual G1 URDF path from Unitree repo
    # For now using placeholder - you'll need to:
    # 1. Download G1 URDF from Unitree repo
    # 2. Convert to USD if needed
    # 3. Point to correct location
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoids/G1/g1.usd",
        activate_contact_sensors=True,
        self_collision=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
            contact_offset=0.02,
            rest_offset=0.001,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),  # Spawn height ~1m (humanoid standing)
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity rotation (quaternion)
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        # TODO: Set initial joint positions if needed
        # joint_pos: Default is all zeros (T-pose)
        # joint_vel: Default is all zeros (at rest)
    ),
    actuators={
        # Arm actuator configuration
        "arm_left": sim_utils.ImplicitActuatorCfg(
            joint_names_expr=["l_.*_joint"],  # Left arm joints regex
            effort_limit=300.0,
            velocity_limit=2.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "arm_right": sim_utils.ImplicitActuatorCfg(
            joint_names_expr=["r_.*_joint"],  # Right arm joints regex
            effort_limit=300.0,
            velocity_limit=2.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # Base/locomotion actuator (if using mobile base)
        "base": sim_utils.ImplicitActuatorCfg(
            joint_names_expr=[".*_hip.*", ".*_knee.*", ".*_ankle.*"],  # Leg joints
            effort_limit=500.0,
            velocity_limit=3.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
)


##
# Gripper Configuration (Dex1 - Two-finger gripper)
##

G1_GRIPPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/gripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Hands/Dex1/dex1.usd",
        activate_contact_sensors=True,
        self_collision=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Attached to robot end-effector
    ),
    actuators={
        "gripper": sim_utils.ImplicitActuatorCfg(
            joint_names_expr=[".*gripper.*"],
            effort_limit=200.0,
            velocity_limit=5.0,
            stiffness=40.0,
            damping=2.0,
        ),
    },
)


##
# Factory Objects (Manipulable items)
##

# Simple cubic object for testing
CUBE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/cube",
    spawn=sim_utils.CubeCfg(
        size=(0.05, 0.05, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.04,
            angular_damping=0.04,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        material_props=sim_utils.RigidBodyMaterialPropertiesCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.8),  # Blue
            opacity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.3, 0.0, 1.1),  # Near robot on table
    ),
)


##
# Factory Environment Elements
##

# Factory workbench/table
TABLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/table",
    spawn=sim_utils.CubeCfg(
        size=(1.0, 1.0, 0.1),  # 1m x 1m x 0.1m table
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            sleep_threshold=0.005,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),  # Heavy table
        material_props=sim_utils.RigidBodyMaterialPropertiesCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.7, 0.7, 0.7),  # Gray
            opacity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Table height ~0.5m
    ),
)
