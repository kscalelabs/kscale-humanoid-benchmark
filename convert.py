"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path
from typing import Callable

import jax
from jaxtyping import Array
from xax.nn.export import export as xax_export

from train import NUM_ACTOR_INPUTS, HumanoidWalkingTask, Model


def make_export_model(model: Model) -> Callable:
    def model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    def batched_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
        return jax.vmap(model_fn)(obs, carry)

    return batched_model_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    model: Model = task.load_ckpt(ckpt_path, part="model")[0]
    model_fn = make_export_model(model)

    input_shapes = [
        (NUM_ACTOR_INPUTS,),
        (
            task.config.depth,
            task.config.hidden_size,
        ),
    ]

    xax_export(model_fn, input_shapes, args.output_path)


if __name__ == "__main__":
    main()
