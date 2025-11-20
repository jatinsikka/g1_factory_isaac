# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""G1 Humanoid Factory Environment for Isaac Lab."""

__version__ = "0.1.0"

# Register environments
try:
    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnv
    from g1_factory_isaac.tasks.factory_env_cfg import FactoryTaskCfg

    # Register the factory environment
    gym.register(
        id="Isaac-FactoryG1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": FactoryTaskCfg,
        },
        disable_env_checker=True,
    )
except ImportError:
    # Isaac Lab or gymnasium not installed yet
    pass
