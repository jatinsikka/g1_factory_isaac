# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""Observation functions for G1 factory manipulation environment."""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_robot_body_joint_states(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get robot body joint states (positions and velocities).
    
    Returns concatenated tensor of [joint_pos, joint_vel] for all 29 DOFs.
    
    Args:
        env: ManagerBasedRLEnv instance
    
    Returns:
        Tensor of shape (num_envs, 58) with [pos_0...pos_28, vel_0...vel_28]
    """
    robot: Articulation = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    
    # Concatenate positions and velocities
    return torch.cat([joint_pos, joint_vel], dim=-1)


def get_gripper_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get gripper state (positions and velocities).
    
    Returns gripper joint positions and velocities.
    Dex3 gripper has 3 fingers, typically 3-4 joints per finger.
    
    Args:
        env: ManagerBasedRLEnv instance
    
    Returns:
        Tensor of shape (num_envs, num_gripper_joints*2)
    """
    gripper: Articulation = env.scene["gripper"]
    gripper_pos = gripper.data.joint_pos
    gripper_vel = gripper.data.joint_vel
    
    return torch.cat([gripper_pos, gripper_vel], dim=-1)


def get_object_relative_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get object position relative to robot end-effector.
    
    Computes relative position of cube to gripper for task guidance.
    
    Args:
        env: ManagerBasedRLEnv instance
    
    Returns:
        Tensor of shape (num_envs, 3) with relative [dx, dy, dz]
    """
    robot: Articulation = env.scene["robot"]
    cube = env.scene["cube"]
    
    # Get end-effector position (approximate as last body or specific link)
    # For G1 with Dex3, the gripper base is at a known link
    robot_pos = robot.data.body_pos_w[:, -1]  # Last body (end-effector)
    cube_pos = cube.data.root_pos_w
    
    # Relative position
    relative_pos = cube_pos - robot_pos
    
    return relative_pos


def get_object_linear_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get object linear velocity.
    
    Observing velocity helps the policy understand dynamics.
    
    Args:
        env: ManagerBasedRLEnv instance
    
    Returns:
        Tensor of shape (num_envs, 3) with velocity [vx, vy, vz]
    """
    cube = env.scene["cube"]
    return cube.data.lin_vel_w


def get_object_position_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get cube position in world frame.
    
    Provides absolute position information.
    
    Args:
        env: ManagerBasedRLEnv instance
    
    Returns:
        Tensor of shape (num_envs, 3) with [x, y, z]
    """
    cube = env.scene["cube"]
    return cube.data.root_pos_w


def get_target_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get target position for cube placement.
    
    This is a constant observation (same for all envs).
    Could be modified for curriculum learning.
    
    Args:
        env: ManagerBasedRLEnv instance
    
    Returns:
        Tensor of shape (num_envs, 3) with target [x, y, z]
    """
    # Default target position on table
    target_pos = torch.tensor(
        [0.5, 0.5, 1.1],
        dtype=torch.float32,
        device=env.device
    )
    
    # Repeat for all environments
    batch_size = env.num_envs
    return target_pos.unsqueeze(0).expand(batch_size, -1)


__all__ = [
    "get_robot_body_joint_states",
    "get_gripper_state",
    "get_object_relative_position",
    "get_object_linear_velocity",
    "get_object_position_world",
    "get_target_position",
]
