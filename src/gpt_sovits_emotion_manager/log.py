import sys
from loguru import logger

from .config import Config


def setup_logger(config: Config):
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<g>{time:MM-DD HH:mm:ss}</g> "
            "[<lvl>{level}</lvl>] "
            # "<c><u>{name}</u></c> | "
            "{message}"
        ),
        level=config.log_level.upper(),
    )
