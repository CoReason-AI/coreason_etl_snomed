# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

import os
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from coreason_etl_snomed.config import EpistemicOntologyPolicy


def test_epistemic_ontology_policy_valid_initialization() -> None:
    """Test valid initialization using explicit arguments."""
    policy = EpistemicOntologyPolicy(
        umls_api_key="valid_key",
        db_uri="postgresql://user:pass@localhost:5432/db",
        bronze_data_path="/custom/path/",
        snomed_namespace_uuid="custom_uuid",
    )
    assert policy.umls_api_key.get_secret_value() == "valid_key"
    assert policy.db_uri.get_secret_value() == "postgresql://user:pass@localhost:5432/db"
    assert policy.bronze_data_path == "/custom/path/"
    assert policy.snomed_namespace_uuid == "custom_uuid"


@patch.dict(
    os.environ,
    {
        "UMLS_API_KEY": "env_api_key",
        "DB_URI": "postgresql://env_user:pass@localhost:5432/db",
    },
    clear=True,
)
def test_epistemic_ontology_policy_env_loading() -> None:
    """Test loading configuration from environment variables."""
    policy = EpistemicOntologyPolicy()
    assert policy.umls_api_key.get_secret_value() == "env_api_key"
    assert policy.db_uri.get_secret_value() == "postgresql://env_user:pass@localhost:5432/db"
    assert policy.bronze_data_path == "data/bronze/snomed/raw/"
    assert policy.snomed_namespace_uuid == "b3d2b3c0-4f5c-4f7f-8e41-0f4b3f1f3e0f"


def test_epistemic_ontology_policy_missing_required() -> None:
    """Test validation failure when required fields are missing."""
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValidationError) as excinfo:
        EpistemicOntologyPolicy()
    errors = excinfo.value.errors()
    missing_fields = [error["loc"][0] for error in errors]
    assert "umls_api_key" in missing_fields
    assert "db_uri" in missing_fields


@given(  # type: ignore
    st.text(min_size=1),
    st.text(min_size=1),
    st.text(min_size=0),
    st.text(min_size=0),
)
def test_epistemic_ontology_policy_hypothesis(
    umls_api_key: str, db_uri: str, bronze_data_path: str, snomed_namespace_uuid: str
) -> None:
    """Property-based testing for configuration initialization."""
    policy = EpistemicOntologyPolicy(
        umls_api_key=umls_api_key,
        db_uri=db_uri,
        bronze_data_path=bronze_data_path,
        snomed_namespace_uuid=snomed_namespace_uuid,
    )
    assert policy.umls_api_key.get_secret_value() == umls_api_key
    assert policy.db_uri.get_secret_value() == db_uri
    assert policy.bronze_data_path == bronze_data_path
    assert policy.snomed_namespace_uuid == snomed_namespace_uuid


@given(st.text(max_size=0))  # type: ignore
def test_epistemic_ontology_policy_empty_secrets(empty_str: str) -> None:
    """Test validation failure when secrets are empty strings."""
    with pytest.raises(ValidationError):
        EpistemicOntologyPolicy(umls_api_key=empty_str, db_uri="valid")

    with pytest.raises(ValidationError):
        EpistemicOntologyPolicy(umls_api_key="valid", db_uri=empty_str)
