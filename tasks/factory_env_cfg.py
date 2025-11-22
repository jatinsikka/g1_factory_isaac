# Copyright (c) 2025, G1 Factory Isaac Lab Project
# All rights reserved.

"""G1 Humanoid Factory Task Environment Configuration."""

import torch

from isaaclab.envs.common import ViewerCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import G1 robot configuration
from ..assets import G1_CFG, G1_GRIPPER_CFG, CUBE_CFG, TABLE_CFG

# Import MDP definitions
from .. import mdp


##
# Scene definition
##

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with G1 humanoid and factory environment."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Factory table/workbench
    table = TABLE_CFG

    # G1 robot
    robot = G1_CFG

    # Gripper attached to robot
    gripper = G1_GRIPPER_CFG

    # Factory objects (parts to manipulate)
    cube = CUBE_CFG

    # TODO: Add more objects as needed
    # cube_2 = CUBE_CFG
    # cylinder = CYLINDER_CFG

    # Lighting (default from scene, can customize)
    # light = AssetBaseCfg(...)


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # TODO: Define task-specific commands
    # Examples:
    # - target_object_pose
    # - target_placement_location
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Arm actions (joint velocity control for both arms)
    arm_action: ActionTerm = ActionTerm(
        asset_name="robot",
        joint_names=[".*_shoulder.*", ".*_elbow.*", ".*_wrist.*"],
        action_type="p_abs",  # Absolute position control
        action_range=(-1.0, 1.0),
        interpolate_scale=(0.1,),  # Smooth interpolation
    )

    # Gripper actions (open/close control)
    gripper_action: ActionTerm = ActionTerm(
        asset_name="gripper",
        joint_names=[".*gripper.*"],
        action_type="p_abs",
        action_range=(-1.0, 1.0),
        interpolate_scale=(0.05,),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot joint states (positions and velocities for all 29 DOFs)
        joint_states = ObsTerm(
            func=mdp.get_robot_body_joint_states,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # Gripper state (positions and velocities for all gripper joints)
        gripper_state = ObsTerm(
            func=mdp.get_gripper_state,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # Object position relative to gripper
        object_relative_pos = ObsTerm(
            func=mdp.get_object_relative_position,
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )

        # Object linear velocity
        object_velocity = ObsTerm(
            func=mdp.get_object_linear_velocity,
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )

        # Target position for cube placement
        target_position = ObsTerm(
            func=mdp.get_target_position,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # TODO: Define reset events as needed
    # Examples:
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_robot_joints,
    #     mode="reset",
    #     interval_range_s=(0, 0),
    # )
    # reset_object_pose = EventTerm(
    #     func=mdp.reset_object_pose,
    #     mode="reset",
    #     interval_range_s=(0, 0),
    # )

    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reaching the object
    object_reached = RewTerm(
        func=mdp.object_reached_reward,
        func_kwargs={"object_cfg": SceneEntityCfg("cube")},
        weight=1.0,
    )

    # Grasping the object
    object_grasped = RewTerm(
        func=mdp.object_grasped_reward,
        weight=2.0,
    )

    # Placing object at target location
    object_placed = RewTerm(
        func=mdp.object_placement_reward,
        func_kwargs={
            "object_cfg": SceneEntityCfg("cube"),
            "target_pos": (0.5, 0.5, 1.1),  # Target location on table
        },
        weight=3.0,
    )

    # Smooth actions (penalty)
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=0.1,
    )

    # Joint velocity penalty (energy efficiency)
    joint_velocity = RewTerm(
        func=mdp.joint_velocity_penalty,
        weight=0.01,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode times out
    time_out = DoneTerm(
        func=lambda env: torch.tensor(
            env.episode_length_buf >= env.max_episode_length,
            device=env.device,
        ).unsqueeze(-1),
        time_out=True,
    )

    # Robot has fallen
    robot_fallen = DoneTerm(
        func=mdp.check_robot_fallen,
        time_out=False,
    )

    # Cube dropped too far
    cube_dropped = DoneTerm(
        func=mdp.check_cube_dropped_far,
        time_out=False,
    )

    # Cube out of bounds
    cube_out_of_bounds = DoneTerm(
        func=mdp.check_cube_out_of_bounds,
        time_out=False,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # TODO: Define curriculum learning strategies if needed
    pass


##
# Environment configuration
##

@configclass
class FactoryTaskCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 factory manipulation environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=256, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg(eye=(1.0, -1.0, 0.5), origin_type="asset_root", asset_name="robot")
    
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = 2
        self.episode_length_s = 30.0  # 30 second episodes
        
        # simulation settings
        self.sim.dt = 0.01
