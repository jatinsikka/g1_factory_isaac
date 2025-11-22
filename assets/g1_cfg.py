# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""G1 Humanoid Robot Asset Configuration.

This module provides pre-configured G1 robot articulation for Isaac Lab.
Adapted from Unitree's isaaclab implementation.

References:
    https://github.com/unitreerobotics/unitree_sim_isaaclab
"""

from pathlib import Path
from isaaclab.assets import ArticulationCfg, RigidBodyCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Get the directory of this file to locate assets
ASSETS_DIR = Path(__file__).parent

##
# G1 Robot Configuration
##

G1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSETS_DIR / "g1.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        link_density=1.0e-8,
        activate_contact_sensors=True,
        self_collision=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Spawn height suitable for humanoid
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity rotation (quaternion)
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    actuators={
        "default": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=10.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)
"""Configuration of G1 humanoid robot using implicit actuator model."""


##
# Gripper Configuration - Using a simple two-finger gripper from Isaac Nucleus
##

G1_GRIPPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/gripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Grippers/Robotiq/2f_140/collision/2f_140.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Attached to robot end-effector
    ),
)


##
# Factory Objects (Manipulable items)
##

# Simple cubic object for testing
CUBE_CFG = RigidBodyCfg(
    prim_path="{ENV_REGEX_NS}/cube",
    spawn=sim_utils.BoxCfg(
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
    init_state=RigidBodyCfg.InitialStateCfg(
        pos=(0.3, 0.0, 0.6),  # Near robot on table
    ),
)


##
# Factory Environment Elements
##

# Factory workbench/table
TABLE_CFG = RigidBodyCfg(
    prim_path="{ENV_REGEX_NS}/table",
    spawn=sim_utils.BoxCfg(
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
    init_state=RigidBodyCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # Table height at 0.05m (0.1/2)
    ),
)
