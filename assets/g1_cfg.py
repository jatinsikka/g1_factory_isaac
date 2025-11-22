# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""G1 Humanoid Robot Asset Configuration.

This module provides pre-configured G1 robot articulation for Isaac Lab.
Adapted from Unitree's isaaclab implementation.

References:
    https://github.com/unitreerobotics/unitree_sim_isaaclab
"""

import os
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Get the absolute path to the assets directory using __file__
ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

##
# G1 Robot Configuration
##

G1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(ASSETS_DIR, "g1.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        link_density=1.0e-8,
        activate_contact_sensors=True,
        self_collision=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=10.0,
                damping=1.0,
            ),
        ),
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
    actuators={
        "default": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=50.0,
            velocity_limit_sim=2.0,
            stiffness=5.0,
            damping=0.5,
            armature=0.005,
        ),
    },
)


##
# Factory Objects (Manipulable items)
##

# Placeholder cube object - will be added later
CUBE_CFG = None
"""Configuration placeholder for cube object."""


##
# Factory Environment Elements
##

# Placeholder table - will be added later
TABLE_CFG = None
"""Configuration placeholder for table."""
