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


def test_epistemic_bronze_extraction_intent_idempotency_and_duplicates(
    mock_policy: EpistemicRxNormPolicy, tmp_path: Path
) -> None:
    """
    Complex Scenario:
    1. Tests that pre-existing files in the bronze directory are safely overwritten (idempotent).
    2. Tests handling of duplicate target files inside the ZIP (e.g. multiple rrf/ folders).
       The extraction should still succeed and the last processed file will overwrite previous ones.
    """
    bronze_dir = Path(mock_policy.bronze_data_path)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate bronze dir with old data
    pre_existing_file = bronze_dir / "RXNCONSO.RRF"
    pre_existing_file.write_text("old data")

    archive_path = tmp_path / "complex_mock.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        # First occurrence of RXNCONSO in an rrf dir
        zf.writestr("folder_a/rrf/RXNCONSO.RRF", b"new data A")
        # Second occurrence of RXNCONSO in another rrf dir
        zf.writestr("folder_b/rrf/RXNCONSO.RRF", b"new data B")
        # Ensure we also test backslashes in ZIP paths (Windows style)
        zf.writestr("folder_c\\rrf\\RXNREL.RRF", b"new rel data")

        # Test backslash in folder but the member path still resolves
        # Need to simulate what zipfile might actually provide with Windows paths
        # Actually, python's zipfile often uses forward slashes regardless, but
        # we added normalization for file_info.filename replacing backslash with forward slash.

    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=archive_path)
    intent.execute()

    # Verify idempotency / overwrite behavior
    assert pre_existing_file.exists()
    assert pre_existing_file.read_text() == "new data B"  # The last one processed should win

    # Verify Windows-style path was normalized and successfully extracted
    rel_file = bronze_dir / "RXNREL.RRF"
    assert rel_file.exists()
    assert rel_file.read_text() == "new rel data"


def test_epistemic_bronze_extraction_intent_write_permission_error(
    mock_policy: EpistemicRxNormPolicy, tmp_path: Path, mocker: MagicMock
) -> None:
    """
    Complex Scenario:
    Simulates a file system error (like PermissionError or Disk Full) when trying to open the target file for writing.
    Ensures that the exception is re-raised and the temp archive is properly cleaned up via the finally block.
    """
    archive_path = tmp_path / "permission_mock.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("rrf/RXNSAT.RRF", b"sat data")

    # Mock Python's built-in open function to raise a PermissionError when writing
    mock_open = mocker.mock_open()
    mock_open.side_effect = PermissionError("Permission denied: cannot write to target path")
    mocker.patch("builtins.open", mock_open)

    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=archive_path)

    with pytest.raises(PermissionError, match="Permission denied"):
        intent.execute()

    # The most critical part of this test: verify the finally block executed and cleaned the file
    assert not archive_path.exists()
