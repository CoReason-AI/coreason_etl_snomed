# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

"""
AGENT INSTRUCTION: This module defines the strict epistemological and configuration boundaries for the SNOMED pipeline.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EpistemicOntologyPolicy(BaseSettings):  # type: ignore
    """
    AGENT INSTRUCTION: Establishes configuration boundaries for the ontology pipeline.
    """

    umls_api_key: SecretStr = Field(
        ...,
        description="The UMLS API key required to access the NLM UTS API for SNOMED CT downloads.",
        min_length=1,
    )
    db_uri: SecretStr = Field(
        ...,
        description="The target PostgreSQL database URI connection string.",
        min_length=1,
    )
    bronze_data_path: str = Field(
        "data/bronze/snomed/raw/",
        description="The local file system path where raw RF2 text files are extracted.",
    )
    snomed_namespace_uuid: str = Field(
        "b3d2b3c0-4f5c-4f7f-8e41-0f4b3f1f3e0f",
        description="The static deterministic UUID string used as the NAMESPACE_SNOMED for UUID5 identity resolution.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
