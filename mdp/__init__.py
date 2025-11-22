# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""MDP definitions for G1 factory environment."""

# Import from submodules
from .rewards import (
    object_reached_reward,
    object_grasped_reward,
    object_placement_reward,
    action_smoothness_penalty,
    joint_velocity_penalty,
)

from .observations import (
    get_robot_body_joint_states,
    get_gripper_state,
    get_object_relative_position,
    get_object_linear_velocity,
    get_object_position_world,
    get_target_position,
)

from .terminations import (
    check_cube_dropped_far,
    check_cube_out_of_bounds,
    check_robot_fallen,
    check_success_reached,
)

__all__ = [
    # Reward functions
    "object_reached_reward",
    "object_grasped_reward",
    "object_placement_reward",
    "action_smoothness_penalty",
    "joint_velocity_penalty",
    # Observation functions
    "get_robot_body_joint_states",
    "get_gripper_state",
    "get_object_relative_position",
    "get_object_linear_velocity",
    "get_object_position_world",
    "get_target_position",
    # Termination functions
    "check_cube_dropped_far",
    "check_cube_out_of_bounds",
    "check_robot_fallen",
    "check_success_reached",
]
