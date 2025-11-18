# logging_config.py
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path.cwd() / "logs"


def setup_logging(
    stdout_level: str = "WARNING",
    parallel: bool = False,
    fname: str = "",
    file_level: str = "TRACE",
    output_dir: Path = LOG_DIR,
):
    logger.remove()
    logger.add(
        sys.stdout,
        level=stdout_level,
        enqueue=True,
    )

    if parallel:
        logger.add(
            output_dir / f"{fname}.log",
            level=file_level,
            rotation="50 MB",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
