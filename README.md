# G1 Humanoid Factory Environment - Isaac Lab

This project implements a reinforcement learning environment for training G1 humanoids to perform factory manipulation tasks using NVIDIA Isaac Lab and Ray for distributed training.

## Project Structure

```
g1_factory_isaac/
├── assets/                  # Robot and object asset configurations
├── tasks/                   # Environment task configurations
├── mdp/                     # Reward/observation/action definitions
├── agents/                  # RL agent configurations (PPO)
├── scripts/
│   ├── train.py            # Main training script
│   ├── play.py             # Policy evaluation/playback
│   ├── local_ray/          # Local Ray configuration
│   │   ├── job_config.yaml
│   │   └── .env.ray        # W&B credentials
│   └── ray/                # Ray cluster utilities
└── README.md
```

## TODO Checklist

### Phase 1: Asset Setup
- [ ] Load G1 robot URDF/USD file
- [ ] Configure factory table/workbench
- [ ] Define manipulable objects (parts)
- [ ] Set up lighting and camera views

### Phase 2: Environment Configuration
- [ ] Implement ActionsCfg (arm, gripper, base actions)
- [ ] Implement ObservationsCfg (joint states, object poses, etc.)
- [ ] Implement CommandsCfg (target poses/locations)
- [ ] Implement RewardsCfg (task-specific rewards)
- [ ] Implement TerminationsCfg (episode termination conditions)

### Phase 3: MDP Functions
- [ ] Define reward calculation functions
- [ ] Define observation computation functions
- [ ] Define action scaling/clipping

### Phase 4: Training
- [ ] Implement train.py main training loop
- [ ] Integrate W&B logging
- [ ] Set up PPO runner configuration
- [ ] Test local training

### Phase 5: Distributed Training (Ray)
- [ ] Update job_config.yaml with correct paths
- [ ] Add W&B credentials to .env.ray
- [ ] Implement Ray job submission script
- [ ] Test on cluster with reduced num_envs
- [ ] Verify W&B logging on cluster

### Phase 6: Policy Evaluation
- [ ] Implement play.py for policy evaluation
- [ ] Add checkpoint loading
- [ ] Add visualization

## Dependencies

- Isaac Lab (https://github.com/isaac-sim/IsaacLab)
- robot_rl (Robot learning framework)
- Ray (Distributed training)
- Weights & Biases (Experiment tracking)

## Quick Start

### Local Testing
```bash
python scripts/train.py --task factory-v0 --num_envs 64 --max_iterations 100
```

### Ray Cluster Training
```bash
# TODO: Implement ray.sh script or use Ray CLI directly
./ray.sh job --task factory-v0 --num_envs 512 --max_iterations 10000 --wandb
```

## Notes

- G1 humanoid has 12 DOF arm + gripper
- Factory scene includes table, parts, target locations
- Training typically requires 1-2 GPU nodes for stable convergence
- Use curriculum learning if initial rewards are sparse

## Resources

- [Isaac Lab Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_lab/index.html)
- [Robot RL Framework](https://github.com/leggedrobotics/rsl_rl)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
