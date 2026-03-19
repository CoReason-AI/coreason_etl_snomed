# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from coreason_etl_rxnorm.config import EpistemicRxNormPolicy
from coreason_etl_rxnorm.extract import EpistemicBronzeExtractionIntent


@pytest.fixture
def mock_policy(tmp_path: Path) -> EpistemicRxNormPolicy:
    """Provides a mock EpistemicRxNormPolicy with a temporary bronze path."""
    os.environ["UMLS_API_KEY"] = "test_key"
    os.environ["DB_URI"] = "postgresql://test:test@localhost:5432/testdb"
    bronze_dir = tmp_path / "data/bronze/rxnorm/raw"
    return EpistemicRxNormPolicy(
        umls_api_key=SecretStr("test_key"),
        db_uri=SecretStr("postgresql://test:test@localhost:5432/testdb"),
        bronze_data_path=str(bronze_dir),
    )


@pytest.fixture
def mock_archive(tmp_path: Path) -> Path:
    """Creates a mock ZIP archive containing RxNorm files and directories."""
    archive_path = tmp_path / "rxnorm_mock.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        # Create an entry that is a directory
        zf.writestr("rxnorm_full_monthly/", "")

        # Target files within the expected directory structure
        zf.writestr("rxnorm_full_monthly/rrf/RXNCONSO.RRF", b"id|concept|active\n1|Aspirin|1")
        zf.writestr("rxnorm_full_monthly/rrf/RXNREL.RRF", b"id|rel|source|destination\n1|isa|1|2")
        zf.writestr("rxnorm_full_monthly/rrf/RXNSAT.RRF", b"id|sat|value\n1|strength|10mg")

        # Files that should be ignored
        zf.writestr("rxnorm_full_monthly/rrf/OTHER_FILE.RRF", b"ignored content")
        zf.writestr("rxnorm_full_monthly/scripts/RXNCONSO.RRF", b"ignored content not in rrf dir")
        zf.writestr("root_level_file.txt", b"ignored content")

    return archive_path


def test_epistemic_bronze_extraction_intent_success(mock_policy: EpistemicRxNormPolicy, mock_archive: Path) -> None:
    """Tests successful extraction of the target files."""
    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=mock_archive)
    intent.execute()

    bronze_dir = Path(mock_policy.bronze_data_path)
    assert bronze_dir.exists()

    expected_files = ["RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF"]
    for file_name in expected_files:
        extracted_file = bronze_dir / file_name
        assert extracted_file.exists()
        assert extracted_file.stat().st_size > 0

    # Ensure archive is cleaned up
    assert not mock_archive.exists()

    # Ensure other files were not extracted
    assert not (bronze_dir / "OTHER_FILE.RRF").exists()
    assert not (bronze_dir / "root_level_file.txt").exists()


def test_epistemic_bronze_extraction_intent_no_target_files(mock_policy: EpistemicRxNormPolicy, tmp_path: Path) -> None:
    """Tests extraction when archive exists but contains no target files."""
    archive_path = tmp_path / "empty_mock.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("some_folder/random.txt", b"data")

    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=archive_path)
    intent.execute()

    bronze_dir = Path(mock_policy.bronze_data_path)
    assert bronze_dir.exists()

    # Should not create any of the target files
    expected_files = ["RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF"]
    for file_name in expected_files:
        assert not (bronze_dir / file_name).exists()

    # Archive should still be cleaned up
    assert not archive_path.exists()


def test_epistemic_bronze_extraction_intent_extraction_failure(
    mock_policy: EpistemicRxNormPolicy, tmp_path: Path, mocker: MagicMock
) -> None:
    """Tests that exceptions during extraction are caught, logged, and re-raised, and archive is cleaned up."""
    # We don't actually need a real zip for this, just a path
    archive_path = tmp_path / "broken_mock.zip"
    archive_path.touch()

    # Force a failure during extraction
    mocker.patch("zipfile.ZipFile", side_effect=zipfile.BadZipFile("File is not a zip file"))

    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=archive_path)

    with pytest.raises(zipfile.BadZipFile):
        intent.execute()

    # Ensure archive is STILL cleaned up even if extraction failed
    assert not archive_path.exists()
