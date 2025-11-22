# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""Termination functions for G1 factory manipulation environment."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def check_cube_dropped_far(env: "ManagerBasedRLEnv", threshold: float = 0.5) -> torch.Tensor:
    """Check if cube has fallen too far from the table.
    
    Terminates episode if cube falls below height threshold.
    
    Args:
        env: ManagerBasedRLEnv instance
        threshold: Height below which cube is considered dropped (meters)
    
    Returns:
        Tensor of shape (num_envs, 1) with bool indicating termination
    """
    cube = env.scene["cube"]
    cube_z = cube.data.root_pos_w[:, 2]  # Get Z height
    
    # Terminate if cube is too low
    done = cube_z < threshold
    
    return done.unsqueeze(-1)


def check_cube_out_of_bounds(
    env: "ManagerBasedRLEnv",
    min_x: float = -1.0,
    max_x: float = 2.0,
    min_y: float = -1.0,
    max_y: float = 2.0,
) -> torch.Tensor:
    """Check if cube has left the working area.
    
    Terminates episode if cube moves outside workspace bounds.
    
    Args:
        env: ManagerBasedRLEnv instance
        min_x, max_x: X bounds
        min_y, max_y: Y bounds
    
    Returns:
        Tensor of shape (num_envs, 1) with bool indicating termination
    """
    cube = env.scene["cube"]
    cube_pos = cube.data.root_pos_w
    
    cube_x = cube_pos[:, 0]
    cube_y = cube_pos[:, 1]
    
    # Check bounds
    in_x = (cube_x >= min_x) & (cube_x <= max_x)
    in_y = (cube_y >= min_y) & (cube_y <= max_y)
    
    # Terminate if out of bounds
    done = ~(in_x & in_y)
    
    return done.unsqueeze(-1)


def check_robot_fallen(env: "ManagerBasedRLEnv", pelvis_height_min: float = 0.3) -> torch.Tensor:
    """Check if robot has fallen over.
    
    Terminates episode if robot's center of mass (pelvis) drops too low.
    
    Args:
        env: ManagerBasedRLEnv instance
        pelvis_height_min: Minimum height of pelvis before considering fallen (meters)
    
    Returns:
        Tensor of shape (num_envs, 1) with bool indicating termination
    """
    robot = env.scene["robot"]
    
    # Get pelvis (root) position
    pelvis_z = robot.data.root_pos_w[:, 2]
    
    # Terminate if pelvis is too low
    done = pelvis_z < pelvis_height_min
    
    return done.unsqueeze(-1)


def check_success_reached(
    env: "ManagerBasedRLEnv",
    target_pos: torch.Tensor,
    distance_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if cube has reached the target position.
    
    Returns True if cube is within distance_threshold of target.
    Can be used as a success termination (optional).
    
    Args:
        env: ManagerBasedRLEnv instance
        target_pos: Target position [x, y, z] shape (3,) or (num_envs, 3)
        distance_threshold: Distance to consider as reached (meters)
    
    Returns:
        Tensor of shape (num_envs, 1) with bool indicating success
    """
    cube = env.scene["cube"]
    cube_pos = cube.data.root_pos_w
    
    # Handle target_pos dimensions
    if target_pos.dim() == 1:
        target_pos = target_pos.unsqueeze(0).expand(env.num_envs, -1)
    
    # Compute distance
    distance = torch.norm(cube_pos - target_pos, dim=-1)
    
    # Success if within threshold
    success = distance < distance_threshold
    
    return success.unsqueeze(-1)


__all__ = [
    "check_cube_dropped_far",
    "check_cube_out_of_bounds",
    "check_robot_fallen",
    "check_success_reached",
]
