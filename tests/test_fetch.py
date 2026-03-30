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
from pathlib import Path

import pytest
import requests
import responses
from pydantic import SecretStr

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.fetch import EpistemicOntologyFetchIntent


@pytest.fixture
def mock_policy() -> EpistemicOntologyPolicy:
    """Provides a mock EpistemicOntologyPolicy."""
    os.environ["UMLS_API_KEY"] = "test_key"
    os.environ["DB_URI"] = "postgresql://test:test@localhost:5432/testdb"
    return EpistemicOntologyPolicy(
        umls_api_key=SecretStr("test_key"),
        db_uri=SecretStr("postgresql://test:test@localhost:5432/testdb"),
        bronze_data_path="data/bronze/snomed/raw/",
    )


@responses.activate
def test_epistemic_ontology_fetch_intent_success(mock_policy: EpistemicOntologyPolicy) -> None:
    """Tests successful execution of the fetch intent."""

    # Mock Release API
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases?releaseType=snomed-ct-us-edition&current=true",
        json=[{"downloadUrl": "https://example.com/snomed.zip"}],
        status=200,
    )

    # Mock Download API
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/download?url=https://example.com/snomed.zip&apiKey=test_key",
        body=b"mock zip content",
        status=200,
        stream=True,
    )

    intent = EpistemicOntologyFetchIntent(policy=mock_policy)
    file_path = intent.execute()

    assert isinstance(file_path, Path)
    assert file_path.exists()
    assert file_path.read_bytes() == b"mock zip content"

    # Cleanup
    file_path.unlink()


@responses.activate
def test_epistemic_ontology_fetch_intent_invalid_release_response(mock_policy: EpistemicOntologyPolicy) -> None:
    """Tests failure when Release API response is invalid."""

    # Mock Release API with invalid JSON
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases?releaseType=snomed-ct-us-edition&current=true",
        json={"error": "not a list"},
        status=200,
    )

    intent = EpistemicOntologyFetchIntent(policy=mock_policy)

    with pytest.raises(ValueError, match=r"Invalid response format from UTS Release API\."):
        intent.execute()


@responses.activate
def test_epistemic_ontology_fetch_intent_download_failure(mock_policy: EpistemicOntologyPolicy) -> None:
    """Tests failure during the file download."""

    # Mock Release API
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases?releaseType=snomed-ct-us-edition&current=true",
        json=[{"downloadUrl": "https://example.com/snomed.zip"}],
        status=200,
    )

    # Mock Download API Failure
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/download?url=https://example.com/snomed.zip&apiKey=test_key",
        status=500,
    )

    intent = EpistemicOntologyFetchIntent(policy=mock_policy)

    with pytest.raises(requests.exceptions.HTTPError, match="500 Server Error"):
        intent.execute()
