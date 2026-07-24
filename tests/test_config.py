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


def test_snomed_configuration_state_valid() -> None:
    """Test standard valid instantiation of the configuration state."""
    with patch.dict(
        os.environ,
        {
            "UMLS_API_KEY": "test_api_key",
            "DB_URI": "postgresql://user:pass@localhost:5432/coreason_db",
        },
    ):
        config = EpistemicOntologyPolicy()
        assert config.umls_api_key.get_secret_value() == "test_api_key"
        assert config.db_uri.get_secret_value() == "postgresql://user:pass@localhost:5432/coreason_db"
        assert config.bronze_data_path == "data/bronze/snomed/raw"
        assert config.log_level == "INFO"


def test_snomed_configuration_state_missing_api_key() -> None:
    """Test failure when UMLS API Key is omitted."""
    with patch.dict(os.environ, {"DB_URI": "postgresql://user:pass@localhost:5432/coreason_db"}, clear=True):
        with pytest.raises(ValidationError) as exc_info:
            EpistemicOntologyPolicy()

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("umls_api_key",) for e in errors)


def test_snomed_configuration_state_missing_db_uri() -> None:
    """Test failure when Database URI is omitted."""
    with patch.dict(os.environ, {"UMLS_API_KEY": "test_api_key"}, clear=True):
        with pytest.raises(ValidationError) as exc_info:
            EpistemicOntologyPolicy()

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("db_uri",) for e in errors)


def test_snomed_configuration_state_invalid_log_level() -> None:
    """Test failure with an invalid logging level constraint."""
    with patch.dict(
        os.environ,
        {
            "UMLS_API_KEY": "test_api_key",
            "DB_URI": "postgresql://user:pass@localhost:5432/coreason_db",
            "LOG_LEVEL": "TRACE",
        },
    ):
        with pytest.raises(ValidationError) as exc_info:
            EpistemicOntologyPolicy()

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("log_level",) for e in errors)


@given(
    api_key=st.text(min_size=1).filter(lambda s: "\x00" not in s),
    db_uri=st.text(min_size=1).filter(lambda s: "\x00" not in s),
)
def test_snomed_configuration_state_property_based(api_key: str, db_uri: str) -> None:
    """Test arbitrary valid string instantiations using hypothesis."""
    with patch.dict(
        os.environ,
        {
            "UMLS_API_KEY": api_key,
            "DB_URI": db_uri,
        },
    ):
        config = EpistemicOntologyPolicy()
        assert config.umls_api_key.get_secret_value() == api_key
        assert config.db_uri.get_secret_value() == db_uri
