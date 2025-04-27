"""Module for deploying joystick-controlled policies on K-Bot."""

import argparse
import asyncio
import logging

import colorlogging

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    logger.info("Args: %s", args)

    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())
