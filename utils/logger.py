from __future__ import annotations

import logging
import sys
import os

_LOGGER = None

def get_logger(
    name: str,
    log_dir: str | None = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    global _LOGGER

    if _LOGGER is not None:
        return _LOGGER
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, "log.txt"),
            mode="a",
            encoding="utf-8"
        )
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGER = logger
    return logger