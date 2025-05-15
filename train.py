"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Self

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

# These are in the order of the neural network outputs.
# Joint name, target position, penalty weight.
ZEROS: list[tuple[str, float, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0, 1.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0), 1.0),
    ("dof_right_shoulder_yaw_02", 0.0, 1.0),
    ("dof_right_elbow_02", math.radians(90.0), 1.0),
    ("dof_right_wrist_00", 0.0, 1.0),
    ("dof_left_shoulder_pitch_03", 0.0, 1.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0), 1.0),
    ("dof_left_shoulder_yaw_02", 0.0, 1.0),
    ("dof_left_elbow_02", math.radians(-90.0), 1.0),
    ("dof_left_wrist_00", 0.0, 1.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0), 1.0),
    ("dof_right_hip_roll_03", math.radians(-0.0), 2.0),
    ("dof_right_hip_yaw_03", 0.0, 2.0),
    ("dof_right_knee_04", math.radians(-50.0), 1.0),
    ("dof_right_ankle_02", math.radians(30.0), 1.0),
    ("dof_left_hip_pitch_04", math.radians(20.0), 1.0),
    ("dof_left_hip_roll_03", math.radians(0.0), 2.0),
    ("dof_left_hip_yaw_03", 0.0, 2.0),
    ("dof_left_knee_04", math.radians(50.0), 1.0),
    ("dof_left_ankle_02", math.radians(-30.0), 1.0),
]


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
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_acc_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU acceleration and gyroscope observations.",
    )
    gait_freq_range: tuple[float, float] = xax.field(
        value=(1.2, 1.5),
        help="The range of gait frequencies to use.",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=100,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=5.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=1.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=1,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


@attrs.define(frozen=True, kw_only=True)
class FeetPhaseReward(ksim.Reward):
    """Tracks swing / stance phasing of the feet."""

    scale: float = 1.0
    feet_pos_obs_name: str = "feet_position_observation"
    joystick_cmd_name: str = "joystick_command"
    gait_freq_cmd_name: str = "gait_frequency_command"
    max_foot_height: float = 0.12
    ctrl_dt: float = 0.02
    sensitivity: float = 0.01
    foot_default_height: float = 0.0

    def _gait_phase(self, phi: Array, swing_height: Array = jnp.array(0.08)) -> Array:
        """Interpolation logic for gait phase.

        https://arxiv.org/pdf/2201.00206
        """
        x = jnp.clip((phi + jnp.pi) / (2 * jnp.pi), 0, 1)
        stance = xax.cubic_bezier_interpolation(jnp.array(0.0), swing_height, 2 * x)
        swing = xax.cubic_bezier_interpolation(swing_height, jnp.array(0.0), 2 * x - 1)
        return jnp.where(x <= 0.5, stance, swing)

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        foot_pos = traj.obs[self.feet_pos_obs_name]
        gait_freq = traj.command[self.gait_freq_cmd_name]

        phase_dt = 2 * jnp.pi * gait_freq * self.ctrl_dt
        steps = jnp.repeat(jnp.int32(traj.timestep / self.ctrl_dt)[:, None], 2, axis=1)

        phase = (
            jnp.fmod(jnp.broadcast_to(jnp.array([0.0, jnp.pi]), steps.shape) + steps * phase_dt + jnp.pi, 2 * jnp.pi)
            - jnp.pi
        )

        ideal_z = self._gait_phase(phase, jnp.array(self.max_foot_height))

        foot_z = jnp.stack([foot_pos[..., 2], foot_pos[..., 5]], axis=-1)
        error = jnp.sum((foot_z - ideal_z) ** 2, axis=-1)
        reward = jnp.exp(-error / self.sensitivity)

        reward *= 1.0 - traj.command[self.joystick_cmd_name][..., 0]  # zero when standing
        return reward


@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(ksim.Reward):
    """Penalises vertical forces above threshold."""

    scale: float = -1.0
    max_contact_force: float = 350.0
    sensor_names: tuple[str, ...]

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        forces = jnp.stack([traj.obs[n] for n in self.sensor_names], axis=-1)
        cost = jnp.clip(jnp.abs(forces[..., 2, :]) - self.max_contact_force, 0)
        return jnp.sum(cost, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalises COM motion while feet are in contact."""

    scale: float = -1.0
    com_vel_obs_name: str = "center_of_mass_velocity_observation"
    feet_contact_obs_name: str = "feet_contact_observation"

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        vel = jnp.linalg.norm(traj.obs[self.com_vel_obs_name][..., :2], axis=-1, keepdims=True)
        contact = traj.obs[self.feet_contact_obs_name]
        return jnp.sum(vel * contact, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v, _ in ZEROS}
        weights = {k: v for k, _, v in ZEROS}
        joint_targets = [zeros[name] for name in names]
        joint_weights = [weights[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            joint_weights=tuple(joint_weights),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
                "dof_left_shoulder_pitch_03",
                "dof_left_shoulder_roll_03",
                "dof_left_shoulder_yaw_02",
                "dof_left_elbow_02",
                "dof_left_wrist_00",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True)
class GaitFrequencyCommand(ksim.Command):
    """Command that holds a (1,) gait-frequency value."""

    gait_freq_lower: float = 1.2
    gait_freq_upper: float = 1.5

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return jax.random.uniform(rng, (1,), minval=self.gait_freq_lower, maxval=self.gait_freq_upper)

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return prev_command


@attrs.define(frozen=True, kw_only=True)
class TimestepPhaseObservation(ksim.TimestepObservation):
    """Observation of the phase of the timestep (matches gait phase calculation in FeetPhaseReward)."""

    ctrl_dt: float = attrs.field(default=0.02)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        gait_freq = state.commands["gait_frequency_command"]
        timestep = super().observe(state, curriculum_level, rng)
        steps = timestep / self.ctrl_dt
        phase_dt = 2 * jnp.pi * gait_freq * self.ctrl_dt
        start_phase = jnp.array([0, jnp.pi])
        phase = start_phase + steps * phase_dt
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi

        # Stand still case
        joystick_cmd = state.commands["joystick_command"]
        # Check if the "stand still" command (index 0 of one-hot encoded vector) is active.
        is_stand_still_command = joystick_cmd[..., 0] == 1.0
        phase = jnp.where(
            is_stand_still_command,
            jnp.array([jnp.pi / 2, jnp.pi]),  # stand still position
            phase,
        )

        return jnp.array([jnp.cos(phase), jnp.sin(phase)]).flatten()


@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(ksim.FeetContactObservation):
    """Flattened (4,) contact flags of both feet."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return super().observe(state, curriculum_level, rng).flatten()


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left_idx: int
    foot_right_idx: int
    floor_threshold: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
    ) -> Self:
        fl = ksim.get_site_data_idx_from_name(physics_model, foot_left_site_name)
        fr = ksim.get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(foot_left_idx=fl, foot_right_idx=fr, floor_threshold=floor_threshold)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        fl = ksim.get_site_pose(state.physics_state.data, self.foot_left_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        fr = ksim.get_site_pose(state.physics_state.data, self.foot_right_idx)[0] + jnp.array(
            [0.0, 0.0, self.floor_threshold]
        )
        return jnp.concatenate([fl, fr], axis=-1)


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
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
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
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(self.num_outputs, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v, _ in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
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
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

        self.num_inputs = num_inputs

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
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
            ksim.FloorFrictionRandomizer.from_geom_name(
                model=physics_model, floor_geom_name="floor", scale_lower=0.1, scale_upper=2.0
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=0.5,
                y_force=0.5,
                z_force=0.3,
                force_range=(0.5, 1.0),
                x_angular_force=0.0,
                y_angular_force=0.0,
                z_angular_force=0.0,
                interval_range=(0.5, 4.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v, _ in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            TimestepPhaseObservation(),
            # ksim.TimestepObservation(),
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(10)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=1.0,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_site_linvel", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_site_angvel", noise=0.0),
            FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=(
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_0",
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_1",
                ),
                foot_right_geom_names=(
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_0",
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_1",
                ),
                floor_geom_names="floor",
            ),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.0,
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.JoystickCommand(),
            GaitFrequencyCommand(
                gait_freq_lower=self.config.gait_freq_range[0],
                gait_freq_upper=self.config.gait_freq_range[1],
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # Standard rewards.
            ksim.StayAliveReward(scale=10.0),
            ksim.JoystickReward(
                forward_speed=2.0,
                backward_speed=1.0,
                strafe_speed=1.0,
                rotation_speed=math.radians(30),
                scale=1.5,
            ),
            ksim.UprightReward(scale=0.5),
            # Normalisation penalties.
            ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.1),
            ksim.JointAccelerationPenalty(scale=-0.01),
            ksim.JointJerkPenalty(scale=-0.01),
            ksim.LinkAccelerationPenalty(scale=-0.01),
            ksim.LinkJerkPenalty(scale=-0.01),
            ksim.AngularVelocityPenalty(index=("x", "y"), scale=-0.005),
            ksim.LinearVelocityPenalty(index=("z",), scale=-0.005),
            # Bespoke rewards.
            BentArmPenalty.create_penalty(physics_model, scale=-0.1),
            StraightLegPenalty.create_penalty(physics_model, scale=-0.2),
            FeetPhaseReward(scale=1.0, foot_default_height=0.04, max_foot_height=0.12),
            FeetSlipPenalty(scale=-0.25),
            ContactForcePenalty(
                scale=-0.03,
                sensor_names=("sensor_observation_left_foot_force", "sensor_observation_right_foot_force"),
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.2),
            ksim.NotUprightTermination(max_radians=math.radians(60)),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        num_joints = len(ZEROS)

        # timestep phase + joint pos / vel + proj_grav
        num_actor_obs = 4 + num_joints * 2 + 3

        if self.config.use_acc_gyro:
            num_actor_obs += 6

        num_commands = 7 + 1  # joystick OHE + gait frequency
        num_actor_inputs = num_actor_obs + num_commands

        num_critic_inputs = (
            num_actor_inputs
            + 4  # feet contact
            + 6  # feet position
            + 3
            + 4  # base pos / quat
            + 138
            + 230  # COM inertia / velocity
            + 3
            + 3  # base linear / angular vel
            + num_joints  # actuator force
            + 3
            + 3  # imu_acc/gyro (privileged copies)
        )

        if self.config.use_acc_gyro:
            num_critic_inputs -= 6

        return Model(
            key,
            num_actor_inputs=num_actor_inputs,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=num_critic_inputs,
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
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
        # time_1 = observations["timestep_observation"]
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        joystick_cmd_ohe_7 = commands["joystick_command"]
        gait_freq_cmd_1 = commands["gait_frequency_command"]

        obs = [
            # jnp.sin(time_1),
            # jnp.cos(time_1),
            timestep_phase_4,  # 4
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
            joystick_cmd_ohe_7,  # 7
            gait_freq_cmd_1,  # 1
        ]
        if self.config.use_acc_gyro:
            obs += [
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
            ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        # time_1 = observations["timestep_observation"]
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        joystick_cmd_ohe_7 = commands["joystick_command"]
        gait_freq_cmd_1 = commands["gait_frequency_command"]

        # privileged obs
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        feet_contact_4 = observations["feet_contact_observation"]
        feet_position_6 = observations["feet_position_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        base_lin_vel_3 = observations["base_linear_velocity_observation"]
        base_ang_vel_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]

        obs_n = jnp.concatenate(
            [
                # jnp.sin(time_1),
                # jnp.cos(time_1),
                timestep_phase_4,  # 4
                joint_pos_n,
                joint_vel_n / 10.0,
                com_inertia_n,
                com_vel_n,
                imu_acc_3,
                imu_gyro_3,
                proj_grav_3,
                actuator_force_n / 100.0,
                base_position_3,
                base_orientation_4,
                base_lin_vel_3,
                base_ang_vel_3,
                feet_contact_4,
                feet_position_6,
                joystick_cmd_ohe_7,
                gait_freq_cmd_1,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _ppo_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
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

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._ppo_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
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
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            global_grad_clip=2.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            # Checkpointing parameters.
            save_every_n_seconds=60,
        ),
    )
