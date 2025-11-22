# G1 Humanoid Factory Manipulation Environment# G1 Humanoid Factory Manipulation Environment# G1 Humanoid Factory Environment - Isaac Lab



Reinforcement learning environment for training G1 humanoid robots to perform factory manipulation tasks using NVIDIA Isaac Lab and Ray for distributed training.



## Overview[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)This project implements a reinforcement learning environment for training G1 humanoids to perform factory manipulation tasks using NVIDIA Isaac Lab and Ray for distributed training.



This project implements a complete RL pipeline with:[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1.0+-green.svg)](https://github.com/isaac-sim/IsaacLab)

- G1 humanoid robot (23 DOFs) with joint position control for arm manipulation

- 52-dimensional observation space (joint states, base velocities)[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)## Project Structure

- Action smoothness penalty reward

- PPO algorithm via RSL-RL

- Distributed training with Ray cluster

- WANDB integration with video recordingA reinforcement learning environment for training humanoid robots (Unitree G1) to perform factory manipulation tasks using NVIDIA Isaac Lab with distributed training via Ray.```



## Project Structureg1_factory_isaac/



```## 📋 Table of Contents├── assets/                  # Robot and object asset configurations

g1_factory_isaac/

├── assets/                 # G1 robot URDF, meshes, configs├── tasks/                   # Environment task configurations

├── tasks/                  # Environment configuration

├── mdp/                    # Reward and observation functions- [Overview](#overview)├── mdp/                     # Reward/observation/action definitions

├── agents/                 # PPO training config

├── scripts/- [Project Structure](#project-structure)├── agents/                  # RL agent configurations (PPO)

│   ├── train.py           # Training script

│   ├── play.py            # Policy evaluation (TODO)- [Implementation Status](#implementation-status)├── scripts/

│   ├── ray.sh             # Ray job submission

│   └── local_ray/         # Ray configuration- [Quick Start](#quick-start)│   ├── train.py            # Main training script

└── README.md

```- [Dependencies](#dependencies)│   ├── play.py             # Policy evaluation/playback



## Implementation Status- [Configuration](#configuration)│   ├── local_ray/          # Local Ray configuration



### Completed- [Training](#training)│   │   ├── job_config.yaml

- G1 robot asset configuration with absolute path resolution

- Actions: 10-DOF arm joint position control- [Results & Monitoring](#results--monitoring)│   │   └── .env.ray        # W&B credentials

- Observations: joint positions/velocities, base linear/angular velocities

- Rewards: action smoothness penalty (-0.01 weight)- [Known Issues](#known-issues)│   └── ray/                # Ray cluster utilities

- Termination: episode timeout (30 seconds)

- Training pipeline with video recording- [Contributing](#contributing)└── README.md

- WANDB integration and logging

- Ray cluster distributed training (2+ GPU nodes)```

- Model checkpointing

## 🎯 Overview

### In Progress

- play.py: Policy evaluation script## TODO Checklist

- Factory scene objects and manipulation rewards

- Robot fallen termination conditionThis project provides a complete reinforcement learning framework for training G1 humanoid robots to manipulate objects in factory settings. It leverages:



### Not Started### Phase 1: Asset Setup

- Gripper control

- Curriculum learning- **NVIDIA Isaac Lab**: Physics simulation and rendering- [ ] Load G1 robot URDF/USD file

- Domain randomization

- **PPO Algorithm**: Proximal Policy Optimization from RSL-RL- [ ] Configure factory table/workbench

## Quick Start

- **Ray Cluster**: Distributed training across multiple nodes- [ ] Define manipulable objects (parts)

### Local Training

```bash- **Weights & Biases**: Experiment tracking and visualization- [ ] Set up lighting and camera views

cd scripts

./isaaclab.sh -p train.py --task Isaac-FactoryG1-v0 --num_envs 64 --max_iterations 100- **Gymnasium**: Standard RL environment interface

```

### Phase 2: Environment Configuration

### Ray Cluster Training

```bash### Key Features- [ ] Implement ActionsCfg (arm, gripper, base actions)

cd scripts

./ray.sh job --task Isaac-FactoryG1-v0 --max_iterations 100 --num_envs 64- [ ] Implement ObservationsCfg (joint states, object poses, etc.)

```

- ✅ G1 humanoid robot with 23 DOFs (actuated arm and base locomotion)- [ ] Implement CommandsCfg (target poses/locations)

Monitor on WANDB: https://wandb.ai/jsikka-the-university-of-texas-at-austin/G1_Factory_Test

- ✅ Joint position control for arm manipulation- [ ] Implement RewardsCfg (task-specific rewards)

## Configuration

- ✅ Distributed training with Ray on GPU clusters- [ ] Implement TerminationsCfg (episode termination conditions)

### Training Parameters (agents/ppo_cfg.py)

- Network: 64x64 MLP (actor/critic)- ✅ WANDB integration for experiment tracking

- Learning rate: 1e-3

- PPO clip: 0.2- ✅ Video recording and upload to WANDB### Phase 3: MDP Functions

- Entropy coef: 0.01

- Video logging every 1000 iterations- ✅ Modular MDP configuration system- [ ] Define reward calculation functions



### Environment Parameters (tasks/factory_env_cfg.py)- 🔄 In-progress: Factory scene with objects and manipulation tasks- [ ] Define observation computation functions

- Parallel environments: 64

- Episode length: 30 seconds- [ ] Define action scaling/clipping

- Physics dt: 0.01s

## 📁 Project Structure

## Dependencies

### Phase 4: Training

- Isaac Lab >= 2.1.0

- robot-rl (PPO and distributed training)```- [ ] Implement train.py main training loop

- Ray >= 2.0.0

- WANDB >= 0.15.0g1_factory_isaac/- [ ] Integrate W&B logging

- PyTorch >= 2.0.0

- Gymnasium >= 0.29.0├── assets/- [ ] Set up PPO runner configuration



## Known Issues & Fixes│   ├── g1.urdf                 # G1 humanoid URDF definition- [ ] Test local training



1. Video recording: Fixed args_cli.video flag preservation through initialization│   ├── g1.usd                  # USD variant for visualization

2. Asset paths on Ray: Fixed with os.path.abspath() in g1_cfg.py

3. WANDB credentials: Fixed by sourcing .env.ray in ray_interface.sh│   ├── g1_cfg.py               # Asset configuration### Phase 5: Distributed Training (Ray)

4. Robot fallen termination: Removed due to dtype mismatch (base_height_l2 returns float, not bool)

│   ├── meshes/                 # Robot mesh files- [ ] Update job_config.yaml with correct paths

## Troubleshooting

│   └── gi.xml                  # Gripper definition (TODO)- [ ] Add W&B credentials to .env.ray

**GPU Memory Error**: Reduce num_envs or gpu_per_worker

```bash├── tasks/- [ ] Implement Ray job submission script

./ray.sh job --task Isaac-FactoryG1-v0 --num_envs 32

```│   ├── factory_env_cfg.py      # Main environment configuration- [ ] Test on cluster with reduced num_envs



**WANDB Not Logging**: Verify API key in scripts/local_ray/.env.ray│   └── __init__.py- [ ] Verify W&B logging on cluster



**Ray Job Fails**: Check cluster status├── mdp/

```bash

ray status│   ├── rewards.py              # Custom reward functions### Phase 6: Policy Evaluation

```

│   ├── __init__.py- [ ] Implement play.py for policy evaluation

## Performance

├── agents/- [ ] Add checkpoint loading

- Single GPU: ~3000 steps/second

- 100 iterations: <1 minute on RTX 5090│   ├── ppo_cfg.py              # PPO training configuration- [ ] Add visualization

- Training verified on 2-node Ray cluster

│   └── __init__.py

## Contributing

├── scripts/## Dependencies

1. Add functions to mdp/ for new observations/rewards/actions

2. Update tasks/factory_env_cfg.py to integrate changes│   ├── train.py                # Main training script

3. Test locally before Ray cluster submission

4. Update README with new status│   ├── play.py                 # Policy evaluation script- Isaac Lab (https://github.com/isaac-sim/IsaacLab)



## License│   ├── ray.sh                  # Ray job submission script- robot_rl (Robot learning framework)



MIT License│   ├── ray_interface.sh        # Ray cluster interface- Ray (Distributed training)



## Acknowledgments│   ├── local_ray/- Weights & Biases (Experiment tracking)



- NVIDIA Isaac Lab framework│   │   ├── .env.ray            # WANDB credentials

- Unitree G1 URDF

- ETH Zurich RSL-RL implementation│   │   └── job_config.yaml     # Ray job configuration## Quick Start

- UT Austin RLGroup guidance

│   └── ray/

│       ├── wrap_resources.py   # Ray resource wrapper### Local Testing

│       ├── task_runner.py      # Ray task runner```bash

│       └── tuner.py            # Ray hyperparameter tunerpython scripts/train.py --task factory-v0 --num_envs 64 --max_iterations 100

├── config/```

│   └── extension.toml          # IsaacLab extension config

├── pyproject.toml              # Project metadata### Ray Cluster Training

├── setup.py                    # Installation script```bash

└── README.md# TODO: Implement ray.sh script or use Ray CLI directly

```./ray.sh job --task factory-v0 --num_envs 512 --max_iterations 10000 --wandb

```

## ✅ Implementation Status

## Notes

### Completed ✓

- G1 humanoid has 12 DOF arm + gripper

#### Phase 1: Asset Setup- Factory scene includes table, parts, target locations

- ✅ G1 robot URDF/USD loaded and configured- Training typically requires 1-2 GPU nodes for stable convergence

- ✅ Asset paths resolved for both local and Ray cluster execution- Use curriculum learning if initial rewards are sparse

- ✅ Articulation and actuator configuration finalized

- ✅ Contact sensors enabled## Resources



#### Phase 2: Environment Configuration- [Isaac Lab Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_lab/index.html)

- ✅ **ActionsCfg**: Joint position control for 10 arm/shoulder joints- [Robot RL Framework](https://github.com/leggedrobotics/rsl_rl)

  - Joint names: `.*_shoulder_pitch_joint`, `.*_shoulder_roll_joint`, `.*_shoulder_yaw_joint`, `.*_elbow_joint`, `.*_wrist_roll_joint`- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)

  - Action scale: 1.0 (direct joint position commands)
- ✅ **ObservationsCfg**: 52-dimensional observation space
  - Joint positions (relative): `joint_pos`
  - Joint velocities (relative): `joint_vel`
  - Base linear velocity: `base_lin_vel`
  - Base angular velocity: `base_ang_vel`
- ✅ **RewardsCfg**: Action smoothness penalty
  - `action_smoothness`: -0.01 weight on action rate L2 norm
- ✅ **TerminationsCfg**: Episode timeout
  - `time_out`: 30-second episodes (terminates after 1500 steps)

#### Phase 3: MDP Functions
- ✅ All observation functions sourced from isaaclab.envs.mdp
- ✅ Reward functions using action rate L2 norm
- ✅ Proper function names and signatures verified
- ✅ Noise configuration for all observations

#### Phase 4: Training Infrastructure
- ✅ **train.py**: Complete training pipeline
  - Environment instantiation with render_mode support
  - Video recording with gym.wrappers.RecordVideo
  - WANDB integration via robot_rl.runners.OnPolicyRunner
  - Automatic model checkpointing
  - WANDB video upload post-training
- ✅ **ppo_cfg.py**: PPO hyperparameters
  - Network: 64×64 actor/critic MLPs
  - Learning rate: 1e-3
  - Entropy coefficient: 0.01
  - Clip parameter: 0.2
  - Video logging enabled at 1000-iteration intervals
- ✅ Environment variables properly sourced in Ray jobs
- ✅ WANDB credentials configured

#### Phase 5: Distributed Training (Ray)
- ✅ Ray job submission working
- ✅ Multi-GPU node support (tested with 2 GPU nodes, RTX 5090s)
- ✅ 64 parallel environment instances
- ✅ WANDB logging from cluster jobs
- ✅ Model checkpointing to Ray storage
- ✅ Video recording enabled during training

#### Phase 6: Monitoring & Logging
- ✅ WANDB dashboard tracking at: https://wandb.ai/jsikka-the-university-of-texas-at-austin/G1_Factory_Test
- ✅ Metrics logged: episode length, episode return, action smoothness
- ✅ Video frames recorded every 1000 iterations
- ✅ Model checkpoints saved per iteration
- ✅ Training logs with timing information

### In Progress 🔄

- 🔄 **play.py**: Policy evaluation script
  - Currently has TODO placeholders
  - Needs: checkpoint loading, non-headless rendering, policy rollout
  
- 🔄 **Factory Scene Objects**: Manipulation targets
  - Object asset definitions (cubes, parts)
  - Object initialization and randomization
  - Collision detection with robot

- 🔄 **Robot Fallen Termination**: Early episode termination
  - Current: time_out termination only
  - Needed: Custom bool-returning termination for base height check
  - Status: Removed due to dtype mismatch (base_height_l2 is float, not bool)

### Not Started ❌

- ❌ **Gripper Control**: End-effector manipulation
  - Gripper asset defined but not integrated
  - Needs: gripper action terms, grasp detection

- ❌ **Manipulation Rewards**: Task-specific objectives
  - Object reaching reward
  - Grasping reward
  - Placement/assembly reward

- ❌ **Curriculum Learning**: Progressive task difficulty
  - Initial policies struggle with complex coordination
  - Consider: spawning object at different distances/heights

- ❌ **Advanced RL Techniques**:
  - Asymmetric actor-critic (critic sees full state)
  - Domain randomization
  - Reward shaping with auxiliary losses

## 🚀 Quick Start

### Prerequisites

```bash
# Must have Isaac Lab environment set up
# Ensure CUDA-capable GPU is available
# Python 3.11+ with isaaclab and ray installed
```

### Local Training (Single GPU)

```bash
cd scripts
./isaaclab.sh -p train.py --task Isaac-FactoryG1-v0 --num_envs 64 --max_iterations 100
```

### Ray Cluster Training (Distributed)

```bash
cd scripts
# Submit job to Ray cluster with 100 iterations
./ray.sh job --task Isaac-FactoryG1-v0 --max_iterations 100 --num_envs 64

# Monitor in WANDB:
# https://wandb.ai/jsikka-the-university-of-texas-at-austin/G1_Factory_Test
```

### Policy Evaluation

```bash
cd scripts
# TODO: Implement and test play.py
# ./isaaclab.sh -p play.py --task Isaac-FactoryG1-v0 --checkpoint logs/.../model_100.pt
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| isaaclab | ≥2.1.0 | Physics simulation and rendering |
| isaaclab-rl | Latest | Isaac Lab RL integration |
| robot-rl | Latest | RSL-RL PPO and distributed training |
| gymnasium | ≥0.29.0 | Standard RL environment API |
| ray | ≥2.0.0 | Distributed training framework |
| wandb | ≥0.15.0 | Experiment tracking |
| torch | ≥2.0.0 | PyTorch for neural networks |
| numpy | Latest | Numerical computations |

### Installation

```bash
# Clone repository
git clone https://github.com/jatinsikka/g1_factory_isaac.git
cd g1_factory_isaac

# Install in development mode
pip install -e .

# Or manually install dependencies
pip install -e .[all]
```

## ⚙️ Configuration

### Training Parameters (`agents/ppo_cfg.py`)

```python
VanillaPPORunnerCfg:
  num_steps_per_env: 24         # Steps per environment per iteration
  max_iterations: 10_100         # Total training iterations
  save_interval: 1_000           # Save checkpoint every 1000 iterations
  learning_rate: 1.0e-3          # Adam optimizer learning rate
  clip_param: 0.2                # PPO clipping parameter
  entropy_coef: 0.01             # Entropy regularization
  gamma: 0.99                    # Discount factor
  lam: 0.95                      # GAE lambda
  log_video: True                # Enable video recording
  video_interval: 1_000          # Record video every 1000 iterations
  video_length: 200              # Frames per video
```

### Environment Parameters (`tasks/factory_env_cfg.py`)

```python
SceneCfg:
  num_envs: 256                 # Number of parallel environments
  env_spacing: 2.5              # Distance between environment copies
  episode_length_s: 30.0        # Episode duration in seconds
  decimation: 2                 # Simulation step ratio (1/decimation)
  sim.dt: 0.01                  # Physics timestep
```

### WANDB Credentials (`.env.ray`)

```bash
WANDB_API_KEY=<your-api-key>
WANDB_USERNAME=jsikka
```

## 📊 Training

### Single GPU Training
- **GPU**: NVIDIA RTX 5090
- **Num Environments**: 64
- **Steps/Second**: ~3000
- **Time per Iteration**: ~0.5 seconds
- **Max Iterations**: 100 (typically completes in <1 minute)

### Multi-GPU Cluster Training
- **Cluster**: Ray with 2 GPU nodes
- **Total GPUs**: 2×RTX 5090
- **Num Environments**: 64 per job
- **Architecture**: 64×64 MLP (actor & critic)
- **Training Status**: ✅ Tested and working

### Training Metrics

The training loop tracks:
- **Episode Length**: Mean timesteps per episode
- **Episode Return**: Cumulative reward per episode
- **Action Smoothness**: L2 norm of action changes
- **Learning Rate**: Current LR for Adam optimizer
- **Policy Loss**: PPO actor loss
- **Value Loss**: Critic regression loss

## 📈 Results & Monitoring

### WANDB Dashboard

All training runs are logged to WANDB for visualization and analysis:

**Project**: https://wandb.ai/jsikka-the-university-of-texas-at-austin/G1_Factory_Test

**Logged Metrics**:
- Training curves for all losses
- Episode metrics (length, return)
- Video recordings from policy rollouts
- Hyperparameter values
- System resource usage (GPU memory, etc.)

### Checkpoint Management

Checkpoints are saved to:
```
logs/g1_factory_test/{timestamp}_{run_name}/
├── model_0.pt          # Initial model
├── model_100.pt        # Model at iteration 100
├── params/
│   ├── env.yaml        # Environment config snapshot
│   └── agent.yaml      # Agent config snapshot
└── videos/train/       # Recorded rollout videos
    ├── rl-video-episode-0.mp4
    └── rl-video-episode-100.mp4
```

## 🐛 Known Issues

### Video Recording Not Working
- **Symptom**: No videos in WANDB dashboard after training
- **Cause**: `args_cli.video` flag being set to False before wrapper initialization
- **Status**: ✅ **FIXED** - Flag now properly preserved for video wrapper
- **Solution**: Updated train.py to keep video flag through initialization

### Asset Path Resolution on Ray
- **Symptom**: `RuntimeError: Failed to find articulation when resolving '/World/envs/env_0/robot'`
- **Cause**: Relative paths not resolving on Ray worker nodes
- **Status**: ✅ **FIXED** - Updated g1_cfg.py to use absolute paths with `os.path.abspath()`
- **Solution**: Changed from `Path(__file__).parent / "g1.urdf"` to `os.path.join(ASSETS_DIR, "g1.urdf")`

### WANDB Environment Variables Not Sourced
- **Symptom**: WANDB credentials not available on Ray jobs
- **Cause**: `.env.ray` not sourced before job submission
- **Status**: ✅ **FIXED** - Added `source $SCRIPT_DIR/.env.ray` in ray_interface.sh
- **Solution**: Modified ray_interface.sh to source credentials before python command

### Robot Fallen Termination Dtype Mismatch
- **Symptom**: `Expected Bool tensor, got Float tensor` error
- **Cause**: `mdp_isaac.base_height_l2` returns float, not bool
- **Status**: ✅ **WORKAROUND** - Removed termination, using time_out only
- **Solution**: Custom termination function needed (marked as TODO)

### No Visualization in Headless Mode
- **Symptom**: No GUI window during training
- **Cause**: Ray cluster runs headless for efficiency
- **Status**: ✅ **EXPECTED** - Designed behavior for cluster training
- **Solution**: Use `play.py` for local visualization with trained checkpoints

## 🔧 Troubleshooting

### Training Fails with GPU Memory Error
```bash
# Reduce parallel environments
./ray.sh job --task Isaac-FactoryG1-v0 --num_envs 32 --max_iterations 100

# Or request lower GPU allocation
./ray.sh job --task Isaac-FactoryG1-v0 --gpu_per_worker 0.5
```

### WANDB Not Logging
```bash
# Verify API key in .env.ray
cat scripts/local_ray/.env.ray

# Check WANDB initialization in logs
grep -i "wandb" logs/g1_factory_test/*/training_output.log
```

### Ray Job Submission Fails
```bash
# Check Ray cluster status
ray status

# Verify cluster has GPUs
ray cluster compute-resource-utilization
```

## 📝 Contributing

When adding new features:

1. **Add to `mdp/`** for observation/reward/action functions
2. **Update `tasks/factory_env_cfg.py`** to integrate into environment
3. **Test locally** first: `./isaaclab.sh -p train.py ...`
4. **Run on Ray** to verify cluster compatibility
5. **Update this README** with new status

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **NVIDIA**: Isaac Lab framework
- **Unitree**: G1 robot URDF
- **ETH Zurich**: RSL-RL PPO implementation
- **UT Austin RLGroup**: Project context and guidance

## 📞 Support & Questions

For issues or questions:
1. Check [Known Issues](#known-issues) section
2. Review WANDB logs for training diagnostics
3. Check Ray cluster status: `ray status`
4. Review Isaac Lab documentation for environment-specific issues

---

**Last Updated**: November 22, 2025  
**Status**: Training pipeline functional and tested on Ray cluster ✅
