# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""Agent configurations for G1 factory environment."""

from .ppo_cfg import VanillaPPORunnerCfg

# Alias for convenience
PPORunnerCfg = VanillaPPORunnerCfg

__all__ = ["VanillaPPORunnerCfg", "PPORunnerCfg"]
