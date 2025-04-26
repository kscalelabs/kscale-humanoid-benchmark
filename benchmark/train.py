"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import math
from dataclasses import dataclass
from typing import Generic, TypeVar

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

NUM_JOINTS = 20
NUM_ACTOR_INPUTS = 51
NUM_CRITIC_INPUTS = 449

MAX_TORQUE = {
    "00": 1.0,
    "02": 14.0,
    "03": 40.0,
    "04": 60.0,
}


@attrs.define(frozen=True)
class GaitFrequencyCommand(ksim.Command):
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


@attrs.define(frozen=True, kw_only=True)
class TimestepPhaseObservation(ksim.TimestepObservation):
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
        return jnp.array([jnp.cos(phase), jnp.sin(phase)]).flatten()


class DefaultHumanoidActor(eqx.Module):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    num_mixtures: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
    ) -> None:
        num_inputs = NUM_ACTOR_INPUTS
        num_outputs = NUM_JOINTS

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs * 3 * num_mixtures,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.num_mixtures = num_mixtures

    def forward(self, obs_n: Array) -> distrax.Distribution:
        prediction_n = self.mlp(obs_n)

        # Splits the predictions into means, standard deviations, and logits.
        slice_len = NUM_JOINTS * self.num_mixtures
        mean_nm = prediction_n[:slice_len].reshape(NUM_JOINTS, self.num_mixtures)
        std_nm = prediction_n[slice_len : slice_len * 2].reshape(NUM_JOINTS, self.num_mixtures)
        logits_nm = prediction_n[slice_len * 2 :].reshape(NUM_JOINTS, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)
        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)
        return dist_n


class DefaultHumanoidCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_CRITIC_INPUTS
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(self, obs_n: Array) -> Array:
        return self.mlp(obs_n)


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
    ) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            num_mixtures=num_mixtures,
        )
        self.critic = DefaultHumanoidCritic(
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

    # Command parameters.
    gait_freq_lower: float = xax.field(
        value=1.25,
        help="The lower bound for the gait frequency.",
    )
    gait_freq_upper: float = xax.field(
        value=1.5,
        help="The upper bound for the gait frequency.",
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

    # Mujoco parameters.
    kp: float = xax.field(
        value=1.0,
        help="The Kp for the actuators",
    )
    kd: float = xax.field(
        value=0.1,
        help="The Kd for the actuators",
    )
    armature: float = xax.field(
        value=1e-2,
        help="A value representing the effective inertia of the actuator armature",
    )
    friction: float = xax.field(
        value=1e-6,
        help="The dynamic friction loss for the actuator",
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
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-v2-feet"), name="robot")
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
                interval_range=(0.25, 0.75),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            TimestepPhaseObservation(),
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
                lag_range=(0.0, 0.1),
            ),
            ksim.ActuatorAccelerationObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
            ksim.TimestepObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            GaitFrequencyCommand(
                gait_freq_lower=self.config.gait_freq_lower,
                gait_freq_upper=self.config.gait_freq_upper,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.StayAliveReward(scale=1.0),
            ksim.AngularVelocityPenalty(index="x", scale=-0.001),
            ksim.AngularVelocityPenalty(index="y", scale=-0.001),
            ksim.NaiveForwardReward(scale=1.0),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.9, unhealthy_z_upper=1.6),
            ksim.PitchTooGreatTermination(max_pitch=math.pi / 3),
            ksim.RollTooGreatTermination(max_roll=math.pi / 3),
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

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
        )

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> None:
        return None

    def run_actor(
        self,
        model: DefaultHumanoidActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Distribution:
        timestep_phase_4 = observations["timestep_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        gait_freq_cmd = commands["gait_frequency_command"]

        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 4
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                gait_freq_cmd,  # 1
            ],
            axis=-1,
        )

        return model.forward(obs_n)

    def run_critic(
        self,
        model: DefaultHumanoidCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        timestep_phase_4 = observations["timestep_phase_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        gait_freq_cmd = commands["gait_frequency_command"]

        obs_n = jnp.concatenate(
            [
                timestep_phase_4,  # 4
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                gait_freq_cmd,  # 1
            ],
            axis=-1,
        )

        return model.forward(obs_n)

    def get_ppo_variables(
        self,
        model: DefaultHumanoidModel,
        trajectory: ksim.Trajectory,
        model_carry: None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, None]:
        # Vectorize over the time dimensions.
        def get_log_prob(transition: ksim.Trajectory) -> Array:
            action_dist_tj = self.run_actor(model.actor, transition.obs, transition.command)
            log_probs_tj = action_dist_tj.log_prob(transition.action)
            assert isinstance(log_probs_tj, Array)
            return log_probs_tj

        log_probs_tj = jax.vmap(get_log_prob)(trajectory)
        assert isinstance(log_probs_tj, Array)

        # Vectorize over the time dimensions.
        values_tj = jax.vmap(self.run_critic, in_axes=(None, 0, 0))(model.critic, trajectory.obs, trajectory.command)

        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs_tj,
            values=values_tj.squeeze(-1),
        )

        return ppo_variables, None

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        model_carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        action_dist_j = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=None, aux_outputs=None)


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking
    # To visualize the environment, use the following command:
    #   python -m examples.walking run_model_viewer=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.walking num_envs=8 rollouts_per_batch=4
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=2,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            # Logging parameters.
            # log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
            max_action_latency=0.01,
            # Checkpointing parameters.
            save_every_n_seconds=60,
        ),
    )
