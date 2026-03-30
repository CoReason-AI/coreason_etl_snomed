# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

import sys
from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["logger"]


# We define a minimal setting class here just to grab LOG_LEVEL
# so we don't circular import EpistemicOntologyPolicy which depends on logger.
class LoggerPolicy(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"


_log_level = LoggerPolicy().log_level

# Remove default handler
logger.remove()

# Sink 1: Stdout (Human-readable)
logger.add(
    sys.stderr,
    level=_log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)

# Ensure logs directory exists
log_path = Path("logs")
if not log_path.exists():  # pragma: no cover
    log_path.mkdir(parents=True, exist_ok=True)

# Sink 2: File (JSON, Rotation, Retention)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    serialize=True,
    enqueue=True,
    level=_log_level,
)
