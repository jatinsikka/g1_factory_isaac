#!/usr/bin/env python3
# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""Policy playback script for G1 Factory task."""

import argparse

# TODO: Import dependencies for policy evaluation
# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab_tasks.manager_based.g1_factory.tasks import FactoryTaskCfg
# from robot_rl.runners import OnPolicyRunner
# import torch


def main():
    """Main playback loop."""
    
    # TODO: Set up argument parser for playback parameters
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="factory-v0")
    # parser.add_argument("--checkpoint", type=str, required=True)
    # parser.add_argument("--num_envs", type=int, default=1)
    # parser.add_argument("--headless", action="store_true")
    # args = parser.parse_args()

    # TODO: Load trained policy
    # checkpoint = torch.load(args.checkpoint)
    # policy = checkpoint["policy"]

    # TODO: Initialize environment
    # env_cfg = FactoryTaskCfg()
    # env_cfg.scene.num_envs = args.num_envs
    # env = ManagerBasedRLEnv(cfg=env_cfg)

    # TODO: Run policy evaluation loop
    # obs, _ = env.reset()
    # while True:
    #     with torch.no_grad():
    #         actions = policy(obs)
    #     obs, rewards, terminated, truncated, info = env.step(actions)

    print("[INFO] Playback script - TODO: Implement policy evaluation")


if __name__ == "__main__":
    main()
