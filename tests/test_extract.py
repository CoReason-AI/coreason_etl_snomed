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
import zipfile
from pathlib import Path

import pytest
from pydantic import SecretStr

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.extract import EpistemicBronzeExtractionIntent


@pytest.fixture
def mock_policy(tmp_path: Path) -> EpistemicOntologyPolicy:
    """Provides a mock EpistemicOntologyPolicy."""
    os.environ["UMLS_API_KEY"] = "test_key"
    os.environ["DB_URI"] = "postgresql://test:test@localhost:5432/testdb"
    return EpistemicOntologyPolicy(
        umls_api_key=SecretStr("test_key"),
        db_uri=SecretStr("postgresql://test:test@localhost:5432/testdb"),
        bronze_data_path=str(tmp_path / "bronze"),
        snomed_namespace_uuid="b3d2b3c0-4f5c-4f7f-8e41-0f4b3f1f3e0f",
    )


def test_epistemic_bronze_extraction_success(tmp_path: Path, mock_policy: EpistemicOntologyPolicy) -> None:
    """Tests successful extraction of valid SNOMED target files."""
    archive_path = tmp_path / "test_snomed.zip"

    # Create a mock zip file
    with zipfile.ZipFile(archive_path, "w") as zip_ref:
        # Valid files
        zip_ref.writestr(
            "SnomedCT_USEditionRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology/sct2_Concept_Snapshot_US1000124_20230901.txt",
            "concept_data",
        )
        zip_ref.writestr(
            "SnomedCT_USEditionRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology/sct2_Description_Snapshot_US1000124_20230901.txt",
            "description_data",
        )
        zip_ref.writestr(
            "SnomedCT_USEditionRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology/sct2_Relationship_Snapshot_US1000124_20230901.txt",
            "relationship_data",
        )

        # Ignored files (wrong directory or wrong prefix)
        zip_ref.writestr(
            "SnomedCT_USEditionRF2_PRODUCTION_20230901T120000Z/Full/Terminology/sct2_Concept_Full_US1000124_20230901.txt",
            "ignored_full_data",
        )
        zip_ref.writestr(
            "SnomedCT_USEditionRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology/sct2_TextDefinition_Snapshot_US1000124_20230901.txt",
            "ignored_text_def",
        )
        zip_ref.writestr(
            "SnomedCT_USEditionRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology/", ""
        )  # Directory entry

    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=archive_path)
    intent.execute()

    bronze_dir = Path(mock_policy.bronze_data_path)

    # Assert target files were extracted
    assert (bronze_dir / "sct2_Concept_Snapshot_US1000124_20230901.txt").read_text() == "concept_data"
    assert (bronze_dir / "sct2_Description_Snapshot_US1000124_20230901.txt").read_text() == "description_data"
    assert (bronze_dir / "sct2_Relationship_Snapshot_US1000124_20230901.txt").read_text() == "relationship_data"

    # Assert ignored files were not extracted
    assert not (bronze_dir / "sct2_Concept_Full_US1000124_20230901.txt").exists()
    assert not (bronze_dir / "sct2_TextDefinition_Snapshot_US1000124_20230901.txt").exists()

    # Assert archive was deleted
    assert not archive_path.exists()


def test_epistemic_bronze_extraction_no_targets(tmp_path: Path, mock_policy: EpistemicOntologyPolicy) -> None:
    """Tests extraction when no target files are found in the archive."""
    archive_path = tmp_path / "test_empty.zip"

    with zipfile.ZipFile(archive_path, "w") as zip_ref:
        zip_ref.writestr("some/random/file.txt", "data")

    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=archive_path)
    intent.execute()

    bronze_dir = Path(mock_policy.bronze_data_path)

    # Bronze dir should be created, but empty
    assert bronze_dir.exists()
    assert len(list(bronze_dir.iterdir())) == 0

    # Assert archive was deleted
    assert not archive_path.exists()


def test_epistemic_bronze_extraction_invalid_archive(tmp_path: Path, mock_policy: EpistemicOntologyPolicy) -> None:
    """Tests extraction failure with an invalid zip file."""
    archive_path = tmp_path / "invalid.zip"
    archive_path.write_text("not a zip file")

    intent = EpistemicBronzeExtractionIntent(policy=mock_policy, archive_path=archive_path)

    with pytest.raises(zipfile.BadZipFile):
        intent.execute()

    # Assert archive was still deleted in the finally block
    assert not archive_path.exists()
