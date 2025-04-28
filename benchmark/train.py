"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Generic, Self, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from xax.nn.export import export

NUM_JOINTS = 20

# joint pos + joint vel + timestep phase + projected_gravity + imu_gyro
OBS_SIZE = 20 * 2 + 4 + 3 + 3

# lin vel + ang vel + gait freq
CMD_SIZE = 2 + 1 + 1
NUM_ACTOR_INPUTS = OBS_SIZE + CMD_SIZE

# Actor inputs + foot contact + foot pos + base pos + base quat
# + base lin vel + base ang vel + actuator frc + base height
NUM_CRITIC_INPUTS = NUM_ACTOR_INPUTS + 2 + 6 + 3 + 4 + 3 + 3 + 20 + 1

# Position targets
NUM_OUTPUTS = NUM_JOINTS * 1

BIASES: list[float] = [
    0.0,  # dof_right_shoulder_pitch_03
    math.radians(-10.0),  # dof_right_shoulder_roll_03
    0.0,  # dof_right_shoulder_yaw_02
    math.radians(90.0),  # dof_right_elbow_02
    0.0,  # dof_right_wrist_00
    0.0,  # dof_left_shoulder_pitch_03
    math.radians(10.0),  # dof_left_shoulder_roll_03
    0.0,  # dof_left_shoulder_yaw_02
    math.radians(-90.0),  # dof_left_elbow_02
    0.0,  # dof_left_wrist_00
    math.radians(-25.0),  # dof_right_hip_pitch_04
    0.0,  # dof_right_hip_roll_03
    0.0,  # dof_right_hip_yaw_03
    math.radians(-50.0),  # dof_right_knee_04
    math.radians(25.0),  # dof_right_ankle_02
    math.radians(25.0),  # dof_left_hip_pitch_04
    0.0,  # dof_left_hip_roll_03
    0.0,  # dof_left_hip_yaw_03
    math.radians(50.0),  # dof_left_knee_04
    math.radians(-25.0),  # dof_left_ankle_02
]


@attrs.define(frozen=True)
class GaitFrequencyCommand(ksim.Command):
    """Command to set the gait frequency of the robot."""

    gait_freq_lower: float = attrs.field(default=1.2)
    gait_freq_upper: float = attrs.field(default=1.5)

    def initial_command(
        self,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Returns (1,) array with gait frequency."""
        return jax.random.uniform(rng, (1,), minval=self.gait_freq_lower, maxval=self.gait_freq_upper)

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return prev_command


@attrs.define(frozen=True)
class AngularVelocityCommand(ksim.Command):
    """Command to turn the robot."""

    scale: float = attrs.field()
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(frozen=True)
class LinearVelocityCommand(ksim.Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_x, rng_y, rng_zero_x, rng_zero_y = jax.random.split(rng, 4)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.uniform(rng_x, (1,), minval=xmin, maxval=xmax)
        y = jax.random.uniform(rng_y, (1,), minval=ymin, maxval=ymax)
        x_zero_mask = jax.random.bernoulli(rng_zero_x, self.x_zero_prob)
        y_zero_mask = jax.random.bernoulli(rng_zero_y, self.y_zero_prob)
        return jnp.concatenate(
            [
                jnp.where(x_zero_mask, 0.0, x),
                jnp.where(y_zero_mask, 0.0, y),
            ]
        )

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[ksim.vis.Marker]:
        return []


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left_idx: int = attrs.field()
    foot_right_idx: int = attrs.field()
    floor_threshold: float = attrs.field(default=0.0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
    ) -> Self:
        foot_left_idx = ksim.get_site_data_idx_from_name(physics_model, foot_left_site_name)
        foot_right_idx = ksim.get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(
            foot_left_idx=foot_left_idx,
            foot_right_idx=foot_right_idx,
            floor_threshold=floor_threshold,
        )

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        foot_left_pos = ksim.get_site_pose(state.physics_state.data, self.foot_left_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        foot_right_pos = ksim.get_site_pose(state.physics_state.data, self.foot_right_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        return jnp.concatenate([foot_left_pos, foot_right_pos], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(ksim.FeetContactObservation):
    """Observation of the feet contact."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        feet_contact_12 = super().observe(state, curriculum_level, rng)
        return feet_contact_12.flatten()


@attrs.define(frozen=True)
class BaseHeightObservation(ksim.Observation):
    """Observation of the base height."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return jnp.atleast_1d(state.physics_state.data.qpos[2])


@attrs.define(frozen=True, kw_only=True)
class TimestepPhaseObservation(ksim.TimestepObservation):
    """Observation of the phase of the timestep."""

    ctrl_dt: float = attrs.field(default=0.02)
    stand_still_threshold: float = attrs.field(default=0.0)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        gait_freq = state.commands["gait_frequency_command"]
        timestep = super().observe(state, curriculum_level, rng)
        steps = timestep / self.ctrl_dt
        phase_dt = 2 * jnp.pi * gait_freq * self.ctrl_dt
        start_phase = jnp.array([0, jnp.pi])  # trotting gait
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # Stand still case
        vel_cmd = state.commands["linear_velocity_command"]
        ang_vel_cmd = state.commands["angular_velocity_command"]
        cmd_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        phase = jnp.where(
            cmd_norm < self.stand_still_threshold,
            jnp.array([jnp.pi / 2, jnp.pi]),  # stand still position
            phase,
        )

        return jnp.array([jnp.cos(phase), jnp.sin(phase)]).flatten()


@attrs.define
class BentArmPenalty(ksim.Reward):
    arm_indices: tuple[int, ...] = attrs.field()
    arm_targets: tuple[float, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos[..., self.arm_indices]
        qpos_targets = jnp.array(self.arm_targets)
        qpos_diff = qpos - qpos_targets
        return xax.get_norm(qpos_diff, "l1").mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: ksim.PhysicsModel,
        scale: float,
        scale_by_curriculum: bool = False,
    ) -> Self:
        joint_names = ksim.get_joint_names_in_order(model)

        names_to_offsets = [
            ("dof_right_shoulder_pitch_03", 0.0),
            ("dof_right_shoulder_roll_03", math.radians(-10.0)),
            ("dof_right_shoulder_yaw_02", 0.0),
            ("dof_right_elbow_02", math.radians(90)),
            ("dof_right_wrist_00", 0.0),
            ("dof_left_shoulder_pitch_03", 0.0),
            ("dof_left_shoulder_roll_03", math.radians(10.0)),
            ("dof_left_shoulder_yaw_02", 0.0),
            ("dof_left_elbow_02", math.radians(-90)),
            ("dof_left_wrist_00", 0.0),
        ]

        arm_indices = [joint_names.index(name) for name, _ in names_to_offsets]
        arm_targets = [offset for _, offset in names_to_offsets]

        return cls(
            arm_indices=tuple(arm_indices),
            arm_targets=tuple(arm_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class FeetPhaseReward(ksim.Reward):
    """Reward for tracking the desired foot height."""

    scale: float = 1.0
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    linear_velocity_cmd_name: str = attrs.field(default="linear_velocity_command")
    angular_velocity_cmd_name: str = attrs.field(default="angular_velocity_command")
    gait_freq_cmd_name: str = attrs.field(default="gait_frequency_command")
    max_foot_height: float = attrs.field(default=0.12)
    ctrl_dt: float = attrs.field(default=0.02)
    sensitivity: float = attrs.field(default=0.01)
    foot_default_height: float = attrs.field(default=0.0)
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.feet_pos_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.feet_pos_obs_name} not found; add it as an observation in your task.")
        if self.gait_freq_cmd_name not in trajectory.command:
            raise ValueError(f"Command {self.gait_freq_cmd_name} not found; add it as a command in your task.")

        # generate phase values
        gait_freq_n = trajectory.command[self.gait_freq_cmd_name]

        phase_dt = 2 * jnp.pi * gait_freq_n * self.ctrl_dt
        steps = jnp.int32(trajectory.timestep / self.ctrl_dt)
        steps = jnp.repeat(steps[:, None], 2, axis=1)

        start_phase = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (steps.shape[0], 2))
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # batch reward over the time dimension
        foot_pos = trajectory.obs[self.feet_pos_obs_name]

        foot_z = jnp.array([foot_pos[..., 2], foot_pos[..., 5]]).T
        ideal_z = self.gait_phase(phase, swing_height=jnp.array(self.max_foot_height))
        error = jnp.sum(jnp.square(foot_z - ideal_z), axis=-1)
        reward = jnp.exp(-error / self.sensitivity)

        # no movement for small velocity command
        vel_cmd = trajectory.command[self.linear_velocity_cmd_name]
        ang_vel_cmd = trajectory.command[self.angular_velocity_cmd_name]
        command_norm = jnp.linalg.norm(jnp.concatenate([vel_cmd, ang_vel_cmd], axis=-1), axis=-1)
        reward *= command_norm > self.stand_still_threshold

        return reward

    def gait_phase(
        self,
        phi: Array | float,
        swing_height: Array = jnp.array(0.08),
    ) -> Array:
        """Interpolation logic for the gait phase.

        Original implementation:
        https://arxiv.org/pdf/2201.00206
        https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/gait.py#L33
        """
        x = (phi + jnp.pi) / (2 * jnp.pi)
        x = jnp.clip(x, 0, 1)
        stance = xax.cubic_bezier_interpolation(jnp.array(0), swing_height, 2 * x)
        swing = xax.cubic_bezier_interpolation(swing_height, jnp.array(0), 2 * x - 1)
        return jnp.where(x <= 0.5, stance, swing)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="sensor_observation_local_linvel_origin")
    command_name: str = attrs.field(default="linear_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")

        command = trajectory.command[self.command_name]
        lin_vel_error = xax.get_norm(command - trajectory.obs[self.linvel_obs_name][..., :2], self.norm).sum(axis=-1)
        reward_value = jnp.exp(-lin_vel_error / self.error_scale)

        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value *= command_norm > self.stand_still_threshold

        return reward_value


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the angular velocity."""

    error_scale: float = attrs.field(default=0.25)
    angvel_obs_name: str = attrs.field(default="sensor_observation_gyro_origin")
    command_name: str = attrs.field(default="angular_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")
    stand_still_threshold: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")

        command = trajectory.command[self.command_name]
        ang_vel_error = jnp.square(command.flatten() - trajectory.obs[self.angvel_obs_name][..., 2])
        reward_value = jnp.exp(-ang_vel_error / self.error_scale)

        command_norm = jnp.linalg.norm(command, axis=-1)
        reward_value *= command_norm > self.stand_still_threshold

        return reward_value


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_ACTOR_INPUTS
        num_outputs = NUM_JOINTS

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = NUM_JOINTS * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(NUM_JOINTS, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(NUM_JOINTS, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(NUM_JOINTS, self.num_mixtures)

        # Add biases to the mean.
        mean_nm = mean_nm + jnp.array(BIASES)[:, None]

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)
        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_CRITIC_INPUTS
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        self.actor = Actor(
            key,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )

    # Checkpoint parameters.
    export_for_inference: bool = xax.field(
        value=False,
        help="Whether to export the model for inference.",
    )

    # Command parameters.
    gait_freq_lower: float = xax.field(
        value=1.2,
        help="The lower bound for the gait frequency.",
    )

    gait_freq_upper: float = xax.field(
        value=1.5,
        help="The upper bound for the gait frequency.",
    )

    stand_still_threshold: float = xax.field(
        value=0.01,
        help="The threshold for standing still.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=10,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=3.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=1.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=50,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingTaskConfig)


class HumanoidWalkingTask(ksim.PPOTask[Config], Generic[Config]):
    def get_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-v2-feet", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot-v2-feet"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        return metadata.joint_name_to_metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.MITPositionActuators(
            physics_model=physics_model,
            joint_name_to_metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.FloorFrictionRandomizer.from_geom_name(physics_model, "floor", scale_lower=0.95, scale_upper=1.05),
            ksim.ArmatureRandomizer(),
            ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "Torso_Side_Right"),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=1.0,
                y_force=1.0,
                z_force=0.0,
                x_angular_force=0.1,
                y_angular_force=0.1,
                z_angular_force=0.3,
                interval_range=(0.5, 4.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            TimestepPhaseObservation(stand_still_threshold=self.config.stand_still_threshold),
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="base_link_quat",
                lag_range=(0.0, 0.5),
            ),
            ksim.ActuatorAccelerationObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="local_linvel_origin", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="gyro_origin", noise=0.0),
            BaseHeightObservation(),
            FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names="KB_D_501L_L_LEG_FOOT_collision_box",
                foot_right_geom_names="KB_D_501R_R_LEG_FOOT_collision_box",
                floor_geom_names="floor",
            ),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.00,
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            LinearVelocityCommand(
                x_range=(-0.3, 0.7),
                y_range=(-0.2, 0.2),
                x_zero_prob=0.1,
                y_zero_prob=0.2,
                switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
            ),
            AngularVelocityCommand(
                scale=0.1,
                zero_prob=0.9,
                switch_prob=self.config.ctrl_dt / 3,  # once per 3 seconds
            ),
            GaitFrequencyCommand(
                gait_freq_lower=self.config.gait_freq_lower,
                gait_freq_upper=self.config.gait_freq_upper,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        ctrl_dt = self.config.ctrl_dt

        return [
            # Standard rewards.
            ksim.StayAliveReward(scale=1.0),
            ksim.UprightReward(index="x", scale=0.1, inverted=False),
            LinearVelocityTrackingReward(
                scale=1.0,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            AngularVelocityTrackingReward(
                scale=0.5,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
            # Normalization penalties (grow with curriculum).
            ksim.ActionSmoothnessPenalty(scale=-0.01),
            ksim.ActuatorForcePenalty(scale=-0.001),
            ksim.BaseJerkZPenalty(ctrl_dt=ctrl_dt, scale=-0.001),
            ksim.LinearVelocityPenalty(index="z", scale=-0.001),
            ksim.AngularVelocityPenalty(index="z", scale=-0.001),
            # Bespoke rewards.
            BentArmPenalty.create(physics_model, scale=-0.1),
            # Phase-based rewards.
            FeetPhaseReward(
                foot_default_height=0.04,
                max_foot_height=0.12,
                scale=2.1,
                stand_still_threshold=self.config.stand_still_threshold,
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.9, unhealthy_z_upper=1.6),
            ksim.PitchTooGreatTermination(max_pitch=math.radians(30)),
            ksim.RollTooGreatTermination(max_roll=math.radians(30)),
            ksim.FastAccelerationTermination(),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
            dt=self.config.ctrl_dt,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        gait_freq_cmd = commands["gait_frequency_command"]

        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 4
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                proj_grav_3,  # 3
                imu_gyro_3,  # 3
                lin_vel_cmd_2,  # 2
                ang_vel_cmd,  # 1
                gait_freq_cmd,  # 1
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd = commands["angular_velocity_command"]
        gait_freq_cmd = commands["gait_frequency_command"]
        # Privileged observations
        feet_contact_2 = observations["feet_contact_observation"]
        feet_position_6 = observations["feet_position_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        base_linear_velocity_3 = observations["base_linear_velocity_observation"]
        base_angular_velocity_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]
        base_height_1 = observations["base_height_observation"]

        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 4
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                proj_grav_3,  # 3
                imu_gyro_3,  # 3
                lin_vel_cmd_2,  # 2
                ang_vel_cmd,  # 1
                gait_freq_cmd,  # 1
                feet_contact_2,  # 2
                feet_position_6,  # 6
                base_position_3,  # 3
                base_orientation_4,  # 4
                base_linear_velocity_3,  # 3
                base_angular_velocity_3,  # 3
                actuator_force_n / 100.0,  # NUM_JOINTS
                base_height_1,  # 1
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        def scan_fn(
            actor_critic_carry: tuple[Array, Array],
            transition: ksim.Trajectory,
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y),
                self.get_initial_model_carry(rng),
                (next_actor_carry, next_critic_carry),
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, trajectory)

        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,
        )

    def make_export_model(self, model: Model) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            dist, carry = model.actor.forward(obs, carry)
            return dist.mode(), carry

        def batched_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
            return jax.vmap(model_fn)(obs, carry)

        return batched_model_fn

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State | None) -> xax.State | None:
        if not self.config.export_for_inference:
            return state

        model: Model = self.load_ckpt(ckpt_path, part="model")[0]

        model_fn = self.make_export_model(model)

        input_shapes = [
            (NUM_ACTOR_INPUTS,),
            (
                self.config.depth,
                self.config.hidden_size,
            ),
        ]

        if state is None:
            tf_path = ckpt_path.parent / "tf_model"
        else:
            tf_path = (
                ckpt_path.parent / "tf_model"
                if self.config.only_save_most_recent
                else ckpt_path.parent / f"tf_model_{state.num_steps}"
            )

        export(
            model_fn,
            input_shapes,
            tf_path,
        )

        return state


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=2,
            epochs_per_log_step=1,
            rollout_length_seconds=10.0,
            render_length_seconds=10.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            max_action_latency=0.01,
            # Checkpointing parameters.
            save_every_n_seconds=60,
            export_for_inference=True,
        ),
    )
