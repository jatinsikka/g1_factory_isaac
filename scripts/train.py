#!/usr/bin/env python3
# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""Training script for G1 factory manipulation environment.

This script trains a policy using PPO to manipulate objects in a factory setting.

Usage:
    ./isaaclab.sh -p train.py --task Isaac-FactoryG1-v0 --num_envs 256
"""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train G1 factory robot policy")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--no-video", action="store_false", dest="video", help="Disable video recording.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-FactoryG1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10000, help="RL Policy training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load from.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Store video recording flag before modifying for app launcher
enable_video_recording = args_cli.video

# always set headless and enable cameras for recording
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
    # Note: we keep args_cli.video = True to use in video wrapper later

# suppress logs
if not hasattr(args_cli, "kit_args"):
    args_cli.kit_args = ""
args_cli.kit_args += " --/log/level=error"
args_cli.kit_args += " --/log/fileLogLevel=error"
args_cli.kit_args += " --/log/outputStreamLevel=error"

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the training script."""

import gymnasium as gym
import torch
from robot_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set torch backend optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def upload_videos_to_wandb(video_dir: str):
    """Upload recorded videos to WANDB.
    
    Args:
        video_dir: Path to directory containing recorded videos
    """
    import glob
    import pathlib
    
    if not os.path.exists(video_dir):
        print(f"[WARNING] Video directory not found: {video_dir}")
        return
    
    # Find all video files (mp4 format)
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    
    if not video_files:
        print(f"[INFO] No videos found in {video_dir}")
        return
    
    print(f"[INFO] Found {len(video_files)} video(s) to upload")
    
    for video_file in video_files:
        try:
            video_name = pathlib.Path(video_file).stem
            print(f"[INFO] Uploading video: {video_name}")
            wandb.log({f"videos/{video_name}": wandb.Video(video_file)})
        except Exception as e:
            print(f"[WARNING] Failed to upload video {video_file}: {e}")


def main():
    """Main training function."""
    
    # Import configs after app launcher is initialized
    # When running on Ray, module is at isaaclab_tasks.manager_based.g1_factory
    # When running locally, module is at g1_factory_isaac
    try:
        from isaaclab_tasks.manager_based.g1_factory.tasks import FactoryTaskCfg
        from isaaclab_tasks.manager_based.g1_factory.agents import PPORunnerCfg
    except ImportError:
        from g1_factory_isaac.tasks import FactoryTaskCfg
        from g1_factory_isaac.agents import PPORunnerCfg
    
    print(f"[INFO] Training configuration:")
    print(f"  Task: {args_cli.task}")
    print(f"  Number of environments: {args_cli.num_envs if args_cli.num_envs else 'default'}")
    print(f"  Maximum iterations: {args_cli.max_iterations}")
    print(f"  Seed: {args_cli.seed}")
    
    # Create environment config
    env_cfg = FactoryTaskCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    # For video recording, reduce number of environments for better visibility
    elif args_cli.video:
        env_cfg.scene.num_envs = 1  # Use single environment for clearer video
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Create agent config
    agent_cfg = PPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    
    # Specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # Create Isaac environment
    print("[INFO] Creating environment...")
    print(f"[INFO] Environment settings: num_envs={env_cfg.scene.num_envs}, render_mode={'rgb_array' if args_cli.video else None}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print(f"[INFO] Environment created successfully. observation_space={env.observation_space}, action_space={env.action_space}")
    
    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print(f"[INFO] Video folder: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Create runner from RSL-RL
    print("[INFO] Creating training runner...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint from {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    
    # Dump configurations to log directory
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg.to_dict())
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # Run training
    print("[INFO] Starting training...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # Upload recorded videos to WANDB if available
    if args_cli.video and WANDB_AVAILABLE:
        print("[INFO] Uploading videos to WANDB...")
        upload_videos_to_wandb(os.path.join(log_dir, "videos", "train"))
    
    # Close the simulator
    env.close()
    print("[INFO] Training completed successfully!")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
