# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

"""Configuration state for SNOMED CT ETL pipeline.

This module provides the central configuration schema for the coreason_etl_snomed
pipeline, leveraging Pydantic Settings to automatically parse configuration from
environment variables and .env files.
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EpistemicOntologyPolicy(BaseSettings):
    """Immutable configuration state for the SNOMED pipeline.

    AGENT INSTRUCTION: Ensure that any modifications to this schema retain strict
    typing and appropriate validation fields according to the project's semantic
    architecture guidelines.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    umls_api_key: SecretStr = Field(
        ...,
        description="The National Library of Medicine (NLM) UMLS API Key for SNOMED downloads.",
        min_length=1,
    )

    bronze_data_path: str = Field(
        "data/bronze/snomed/raw",
        description="The target local directory for extracting raw Bronze RF2 textual snapshot files.",
    )

    db_uri: SecretStr = Field(
        ...,
        description="The PostgreSQL SQLAlchemy-compatible connection string for the CoReason Knowledge Graph.",
    )

    snomed_namespace_uuid: str = Field(
        "b3d2b3c0-4f5c-4f7f-8e41-0f4b3f1f3e0f", description="The UUID5 namespace for identity generation."
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="The operating log level for the pipeline.",
    )
