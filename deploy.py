"""Module for deploying joystick-controlled policies on K-Bot."""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import colorlogging
import numpy as np
import pykos
import tensorflow as tf
from xax.nn.geom import rotate_vector_by_quat

logger = logging.getLogger(__name__)

RunMode = Literal["real", "sim"]

MAX_TORQUE = {
    "00": 1.0,  # 00 motor
    "02": 13.0,  # 02 motor
    "03": 48.0,  # 03 motor
    "04": 96.0,  # 04 motor
}


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int  # nn_id is the index of the actuator in the neural network output
    kp: float
    kd: float
    max_torque: float
    joint_name: str


actuator_list: list[Actuator] = [
    # Right arm (nn_id 0-4)
    Actuator(
        actuator_id=21, nn_id=0, kp=100.0, kd=4.0, max_torque=MAX_TORQUE["03"], joint_name="dof_right_shoulder_pitch_03"
    ),
    Actuator(
        actuator_id=22, nn_id=1, kp=100.0, kd=4.0, max_torque=MAX_TORQUE["03"], joint_name="dof_right_shoulder_roll_03"
    ),
    Actuator(
        actuator_id=23, nn_id=2, kp=40.0, kd=2.0, max_torque=MAX_TORQUE["02"], joint_name="dof_right_shoulder_yaw_02"
    ),
    Actuator(actuator_id=24, nn_id=3, kp=40.0, kd=2.0, max_torque=MAX_TORQUE["02"], joint_name="dof_right_elbow_02"),
    Actuator(actuator_id=25, nn_id=4, kp=20.0, kd=2.0, max_torque=MAX_TORQUE["00"], joint_name="dof_right_wrist_00"),
    # Left arm (nn_id 5-9)
    Actuator(
        actuator_id=11, nn_id=5, kp=100.0, kd=4.0, max_torque=MAX_TORQUE["03"], joint_name="dof_left_shoulder_pitch_03"
    ),
    Actuator(
        actuator_id=12, nn_id=6, kp=100.0, kd=4.0, max_torque=MAX_TORQUE["03"], joint_name="dof_left_shoulder_roll_03"
    ),
    Actuator(
        actuator_id=13, nn_id=7, kp=40.0, kd=2.0, max_torque=MAX_TORQUE["02"], joint_name="dof_left_shoulder_yaw_02"
    ),
    Actuator(actuator_id=14, nn_id=8, kp=40.0, kd=2.0, max_torque=MAX_TORQUE["02"], joint_name="dof_left_elbow_02"),
    Actuator(actuator_id=15, nn_id=9, kp=20.0, kd=2.0, max_torque=MAX_TORQUE["00"], joint_name="dof_left_wrist_00"),
    # Right leg (nn_id 10-14)
    Actuator(
        actuator_id=41, nn_id=10, kp=150.0, kd=8.0, max_torque=MAX_TORQUE["04"], joint_name="dof_right_hip_pitch_04"
    ),
    Actuator(
        actuator_id=42, nn_id=11, kp=200.0, kd=8.0, max_torque=MAX_TORQUE["03"], joint_name="dof_right_hip_roll_03"
    ),
    Actuator(
        actuator_id=43, nn_id=12, kp=100.0, kd=4.0, max_torque=MAX_TORQUE["03"], joint_name="dof_right_hip_yaw_03"
    ),
    Actuator(actuator_id=44, nn_id=13, kp=150.0, kd=8.0, max_torque=MAX_TORQUE["04"], joint_name="dof_right_knee_04"),
    Actuator(actuator_id=45, nn_id=14, kp=40.0, kd=8.0, max_torque=MAX_TORQUE["02"], joint_name="dof_right_ankle_02"),
    # Left leg (nn_id 15-19)
    Actuator(
        actuator_id=31, nn_id=15, kp=150.0, kd=8.0, max_torque=MAX_TORQUE["04"], joint_name="dof_left_hip_pitch_04"
    ),
    Actuator(
        actuator_id=32, nn_id=16, kp=200.0, kd=8.0, max_torque=MAX_TORQUE["03"], joint_name="dof_left_hip_roll_03"
    ),
    Actuator(actuator_id=33, nn_id=17, kp=100.0, kd=4.0, max_torque=MAX_TORQUE["03"], joint_name="dof_left_hip_yaw_03"),
    Actuator(actuator_id=34, nn_id=18, kp=150.0, kd=8.0, max_torque=MAX_TORQUE["04"], joint_name="dof_left_knee_04"),
    Actuator(actuator_id=35, nn_id=19, kp=40.0, kd=8.0, max_torque=MAX_TORQUE["02"], joint_name="dof_left_ankle_02"),
]

home_position = {
    21: 0.0,  # dof_right_shoulder_pitch_03
    22: -10.0,  # dof_right_shoulder_roll_03
    23: 0.0,  # dof_right_shoulder_yaw_02
    24: 90.0,  # dof_right_elbow_02
    25: 0.0,  # dof_right_wrist_00
    11: 0.0,  # dof_left_shoulder_pitch_03
    12: 10.0,  # dof_left_shoulder_roll_03
    13: 0.0,  # dof_left_shoulder_yaw_02
    14: -90.0,  # dof_left_elbow_02
    15: 0.0,  # dof_left_wrist_00
    41: -25.0,  # dof_right_hip_pitch_04
    42: 0.0,  # dof_right_hip_roll_03
    43: 0.0,  # dof_right_hip_yaw_03
    44: -50.0,  # dof_right_knee_04
    45: 25.0,  # dof_right_ankle_02
    31: 25.0,  # dof_left_hip_pitch_04
    32: 0.0,  # dof_left_hip_roll_03
    33: 0.0,  # dof_left_hip_yaw_03
    34: 50.0,  # dof_left_knee_04
    35: -25.0,  # dof_left_ankle_02
}


@dataclass
class DeployConfig:
    model_path: str = field(default="", metadata={"help": "Path to the model to deploy"})
    action_scale: float = field(default=0.1, metadata={"help": "Scale of the action outputs"})
    run_mode: RunMode = field(default="sim", metadata={"help": "Run mode"})
    joystick_enabled: bool = field(default=False, metadata={"help": "Whether to use joystick"})
    episode_length: int = field(default=10, metadata={"help": "Length of the episode to run in seconds"})
    ip: str = field(default="localhost", metadata={"help": "KOS server IP address"})
    port: int = field(default=50051, metadata={"help": "KOS server port"})
    # Logging
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode"})
    log_dir: str = field(default="rollouts", metadata={"help": "Directory to save rollouts"})
    save_plots: bool = field(default=False, metadata={"help": "Whether to save plots"})
    # Policy parameters
    gait: float = field(default=1.25, metadata={"help": "Gait of the policy"})
    dt: float = field(default=0.02, metadata={"help": "Timestep of the policy"})
    rnn_carry_shape: tuple[int, int] = field(
        default=(5, 128), metadata={"help": "Shape of the RNN carry. (num_layers, hidden_size)"}
    )

    def __repr__(self) -> str:
        return "DeployConfig(\n" + "\n".join([f"  {k}={v}" for k, v in self.__dict__.items()]) + "\n)"

    def to_dict(self) -> dict:
        return self.__dict__


StepDataDictableKey: TypeAlias = Literal["obs", "cmd"]


class StepDataDict(TypedDict):
    action: list[float]
    obs: dict[str, list[float]]
    cmd: dict[str, list[float]]


class HeaderDict(TypedDict):
    units: dict[str, dict[str, str]]
    config: dict[str, dict]
    actuator_config: list[dict]
    home_position: dict[int, float]
    date: str


class RolloutDict(TypedDict):
    header: HeaderDict
    data: dict[str, StepDataDict]


async def run_policy(config: DeployConfig) -> None:
    phase = np.array([0, np.pi])

    async def get_obs(kos_client: pykos.KOS) -> dict:
        nonlocal phase
        actuator_states, imu, quaternion = await asyncio.gather(
            kos_client.actuator.get_actuators_state([ac.actuator_id for ac in actuator_list]),
            kos_client.imu.get_imu_values(),
            kos_client.imu.get_quaternion(),
        )

        # Joint observations
        sorted_actuator_list = sorted(actuator_list, key=lambda x: x.nn_id)

        state_dict_pos = {state.actuator_id: state.position for state in actuator_states.states}
        pos_obs = [state_dict_pos[ac.actuator_id] for ac in sorted_actuator_list]
        pos_obs = np.deg2rad(np.array(pos_obs))  # PyKOS returns degrees, the model expects radians

        state_dict_vel = {state.actuator_id: state.velocity for state in actuator_states.states}
        vel_obs = np.deg2rad(np.array([state_dict_vel[ac.actuator_id] for ac in sorted_actuator_list]))

        # IMU observations
        imu_gyro = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z])
        projected_gravity = rotate_vector_by_quat(
            np.array([0, 0, -9.81]),  # type: ignore[arg-type]
            np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z]),  # type: ignore[arg-type]
            inverse=True,
        )

        # Timestep phase
        phase += 2 * np.pi * config.gait * config.dt
        phase = np.fmod(phase + np.pi, 2 * np.pi) - np.pi
        phase_vec = np.array([np.cos(phase), np.sin(phase)]).flatten()

        return {
            "timestep_phase": phase_vec,
            "pos_obs": pos_obs,
            "vel_obs": vel_obs,
            "imu_gyro": imu_gyro,
            "projected_gravity": projected_gravity,
        }

    def obs_to_vec(obs: dict, cmd: dict) -> np.ndarray:
        # Modify this as needed to match your model's input format
        return np.concatenate(
            [
                obs["pos_obs"],
                obs["vel_obs"],
                obs["projected_gravity"],
            ]
        )[None, :]

    async def get_command(joystick_enabled: bool) -> dict:
        if joystick_enabled:
            raise NotImplementedError
        else:
            return {
                "linear_command_x": np.array([0.0]),
                "linear_command_y": np.array([0.0]),
                "angular_command": np.array([0.0]),
            }

    async def send_action(raw_action: np.ndarray, kos_client: pykos.KOS) -> None:
        raw_position = raw_action[: len(actuator_list)]

        position_target = np.rad2deg(raw_position) * config.action_scale

        actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": ac.actuator_id,
                "position": position_target[ac.nn_id],
            }
            for ac in actuator_list
        ]

        await kos_client.actuator.command_actuators(actuator_commands)

    async def disable(kos_client: pykos.KOS) -> None:
        for ac in actuator_list:
            await kos_client.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=False,
                max_torque=ac.max_torque,
            )

    async def enable(kos_client: pykos.KOS) -> None:
        for ac in actuator_list:
            await kos_client.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=True,
                max_torque=ac.max_torque,
            )

    async def go_home(kos_client: pykos.KOS) -> None:
        actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": id,
                "position": position,
            }
            for id, position in home_position.items()
        ]

        await kos_client.actuator.command_actuators(actuator_commands)

    async def reset_sim(kos_client: pykos.KOS) -> None:
        logger.info("Resetting simulation...")
        await kos_client.sim.reset(pos={"x": 0.0, "y": 0.0, "z": 1.01}, quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})

    async def preflight() -> None:
        os.makedirs(Path(config.log_dir) / config.run_mode, exist_ok=True)
        logger.info("Enabling motors...")
        await enable(kos_client)

        logger.info("Moving to home position...")
        await go_home(kos_client)

    async def postflight() -> None:
        datetime_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the directory for this specific rollout
        rollout_dir = Path(config.log_dir) / config.run_mode / datetime_name
        rollout_dir.mkdir(parents=True, exist_ok=True)

        rollout_file = rollout_dir / f"rollout_{datetime_name}.json"
        logger.info("Saving rollout data to %s", rollout_file)
        with open(rollout_file, "w") as f:
            json.dump(rollout_dict, f, indent=2)

        logger.info("Rollout data saved to %s", rollout_file)

        logger.info("Disabling motors...")
        await disable(kos_client)

        logger.info("Motors disabled")

        if config.save_plots:
            logger.info("Saving plots...")
            import matplotlib.pyplot as plt

            timestamps = [float(t) for t in rollout_dict["data"].keys()]
            data_values: list[StepDataDict] = list(rollout_dict["data"].values())

            plot_dir = rollout_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            def save_plot(
                filename_suffix: str, title: str, data_dict: StepDataDictableKey, labels: dict[str, str]
            ) -> None:
                plt.figure(figsize=(12, 6))
                for key, label in labels.items():
                    y_data = np.array([d[data_dict][key] for d in data_values])
                    if y_data.ndim == 1:
                        plt.plot(timestamps, y_data, label=label)
                    else:
                        for i in range(y_data.shape[1]):
                            plt.plot(timestamps, y_data[:, i], label=f"{label}_{i}")

                plt.title(title)
                plt.xlabel("Time (s)")
                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                plt.grid(True)
                plt.tight_layout(rect=(0, 0, 0.9, 1))
                plot_path = plot_dir / f"{filename_suffix}_{datetime_name}.png"
                plt.savefig(plot_path)
                plt.close()
                logger.info("Plot saved to %s", plot_path)

            # Plot Observations
            save_plot("obs_pos", "Observed Joint Positions (rad)", "obs", {"pos_obs": "Position"})
            save_plot("obs_vel", "Observed Joint Velocities (rad/s)", "obs", {"vel_obs": "Velocity"})
            save_plot(
                "obs_imu",
                "Observed IMU Data",
                "obs",
                {"imu_gyro": "Gyro (rad/s)", "projected_gravity": "Gravity (m/s^2)"},
            )
            save_plot("obs_phase", "Observed Timestep Phase", "obs", {"timestep_phase": "Phase (cos/sin)"})

            # Plot Commands
            save_plot(
                "cmd",
                "Commands",
                "cmd",
                {
                    "linear_command_x": "Linear X (m/s)",
                    "linear_command_y": "Linear Y (m/s)",
                    "angular_command": "Angular (rad/s)",
                },
            )

            num_joints = len(actuator_list)

            action_data = np.array([d["action"] for d in data_values])
            pos_action = action_data[:, :num_joints]

            plt.figure(figsize=(12, 6))
            for i in range(pos_action.shape[1]):
                plt.plot(timestamps, pos_action[:, i], label=f"Pos Action Joint_{i}")
            plt.title("Action - Position Targets (rad)")
            plt.xlabel("Time (s)")
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout(rect=(0, 0, 0.9, 1))
            plot_path = plot_dir / f"action_pos_{datetime_name}.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info("Plot saved to %s", plot_path)

    rollout_dict: RolloutDict = {
        "header": {
            "units": {
                "obs": {
                    "Projected gravity": "Units in m/s^2",
                    "IMU gyro": "Units in rad/s",
                    "Timestep phase": "",
                    "Position": "Units in rad",
                    "Velocity": "Units in rad/s",
                },
                "cmd": {
                    "Linear command": "Units in m/s",
                    "Angular command": "Units in rad/s",
                },
                "action": {
                    "Position": "Units in rad",
                },
            },
            "config": config.to_dict(),
            "actuator_config": [ac.__dict__ for ac in actuator_list],
            "home_position": home_position,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": {},
    }

    kos_client = pykos.KOS(ip=config.ip, port=config.port)

    model = tf.saved_model.load(config.model_path)

    # Warm up model
    logger.info("Warming up model...")
    obs = await get_obs(kos_client)
    cmd = await get_command(config.joystick_enabled)
    carry = np.zeros(config.rnn_carry_shape)[None, :]
    _ = model.infer(obs_to_vec(obs, cmd), carry)

    logger.info("Starting preflight...")
    await preflight()

    if config.run_mode == "real":
        for i in range(5, -1, -1):
            logger.info("Starting rollout in %d...", i)
    else:
        await reset_sim(kos_client)

    start_time = time.time()
    target_time = start_time + config.dt

    try:
        while time.time() - start_time < config.episode_length:
            action, carry = model.infer(obs_to_vec(obs, cmd), carry)

            action_array = np.array(action).reshape(-1)

            elapsed_time = time.time() - start_time
            rollout_dict["data"][f"{elapsed_time:.4f}"] = StepDataDict(
                obs={k: v.tolist() for k, v in obs.items()},
                cmd={k: v.tolist() for k, v in cmd.items()},
                action=action_array.tolist(),
            )

            obs, cmd, _ = await asyncio.gather(
                get_obs(kos_client),
                get_command(config.joystick_enabled),
                send_action(action_array, kos_client),
            )

            if time.time() > target_time:
                logger.warning("Loop overran by %f seconds", time.time() - target_time)
            else:
                logger.debug("Sleeping for %f seconds", target_time - time.time())
                await asyncio.sleep(max(0, target_time - time.time()))

            target_time += config.dt

    except asyncio.CancelledError:
        logger.info("Episode cancelled")

    await postflight()


# python benchmark/deploy.py converted --action-scale 1.0
async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--action-scale", type=float, default=0.1)
    parser.add_argument("--run-mode", type=str, default="sim")
    parser.add_argument("--joystick-enabled", action="store_true")
    parser.add_argument("--episode-length", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--log-dir", type=str, default="rollouts")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    config = DeployConfig(**vars(args))

    logger.info("Args: %s", config)

    await run_policy(config)


if __name__ == "__main__":
    asyncio.run(main())
