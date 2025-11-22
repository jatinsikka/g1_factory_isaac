# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""MDP functions for G1 factory manipulation environment.

This module provides reward, observation, and action computation functions
for the factory manipulation task.
"""

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandsManager, SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


##
# Reward Functions
##

def object_reached_reward(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for gripper proximity to object.
    
    Higher reward when gripper is close to target object.
    Encourages the robot to reach and grasp objects.
    """
    object = env.scene[object_cfg.name]
    # Get gripper position (typically end-effector)
    robot: Articulation = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, -1]  # Last body is end-effector (approximate)
    object_pos = object.data.root_pos_w
    
    distance = torch.norm(ee_pos - object_pos, dim=-1)
    # Reward decreases with distance, max reward at 0 distance
    reward = torch.exp(-3.0 * distance)  # Exponential decay
    
    return reward


def object_grasped_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for successfully grasping object.
    
    Rewards when gripper is close to object (as proxy for grasping).
    """
    object_pos = env.scene["cube"].data.root_pos_w
    robot = env.scene["robot"]
    
    # Get gripper position (last body is end-effector)
    ee_pos = robot.data.body_pos_w[:, -1]
    
    # Distance to object
    distance = torch.norm(ee_pos - object_pos, dim=-1)
    
    # Large reward when very close (grasping range, ~0.05m)
    # Exponential with tighter threshold
    reward = torch.exp(-15.0 * distance)
    
    return reward


def object_placement_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg,
    target_pos: torch.Tensor
) -> torch.Tensor:
    """Reward for placing object at target location.
    
    Args:
        env: Environment instance
        object_cfg: Configuration for the object to place
        target_pos: Target position for object placement
    """
    object = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w
    
    distance = torch.norm(object_pos - target_pos, dim=-1)
    # Large reward when object reaches target (within threshold)
    threshold = 0.1  # 10cm
    reward = (distance < threshold).float()
    
    return reward


def action_smoothness_penalty(actions: torch.Tensor) -> torch.Tensor:
    """Penalty for non-smooth actions.
    
    Discourages jerky movements and encourages smooth trajectories.
    """
    # L2 norm of actions (smaller is smoother)
    return -torch.norm(actions, dim=-1) * 0.01


def joint_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high joint velocities.
    
    Encourages energy-efficient movements.
    """
    robot: Articulation = env.scene["robot"]
    joint_vel = robot.data.joint_vel
    
    # Penalize high velocities
    return -torch.sum(joint_vel ** 2, dim=-1) * 0.001


##
# Observation Functions
##

def object_pose_obs(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Relative position of object in gripper frame.
    
    Returns 3D position (x, y, z) of object relative to gripper.
    """
    object = env.scene[object_cfg.name]
    robot: Articulation = env.scene["robot"]
    
    # Get gripper (end-effector) position
    ee_pos = robot.data.body_pos_w[:, -1]  # Last body
    object_pos = object.data.root_pos_w
    
    # Relative position
    relative_pos = object_pos - ee_pos
    
    return relative_pos


def robot_joint_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Robot joint positions and velocities.
    
    Returns concatenated tensor of [joint_pos, joint_vel]
    """
    robot: Articulation = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    
    return torch.cat([joint_pos, joint_vel], dim=-1)


def gripper_state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gripper state (opening position and velocity).
    
    Returns gripper joint positions and velocities.
    """
    gripper: Articulation = env.scene["gripper"]
    gripper_pos = gripper.data.joint_pos
    gripper_vel = gripper.data.joint_vel
    
    return torch.cat([gripper_pos, gripper_vel], dim=-1)


##
# Action Functions
##

def arm_action_scale(actions: torch.Tensor) -> torch.Tensor:
    """Scale arm actions to joint velocity commands.
    
    Scales [-1, 1] action range to actual joint velocity limits.
    """
    max_vel = 2.0  # rad/s
    return actions * max_vel


def gripper_action_scale(actions: torch.Tensor) -> torch.Tensor:
    """Scale gripper actions to open/close commands.
    
    Actions: [-1, 1]
    -1: fully closed
    0: neutral
    1: fully open
    """
    # Map to joint velocity or position target
    return actions * 5.0  # Scale to rad/s for velocity control
