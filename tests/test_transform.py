# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

from pathlib import Path

import polars as pl
import pytest

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.transform import EpistemicSilverConceptIntent


@pytest.fixture
def policy(tmp_path: Path) -> EpistemicOntologyPolicy:
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    return EpistemicOntologyPolicy(
        umls_api_key="fake",
        db_uri="postgresql://user:pass@localhost:5432/db",
        bronze_data_path=str(bronze_dir),
        snomed_namespace_uuid="b3d2b3c0-4f5c-4f7f-8e41-0f4b3f1f3e0f",
    )


def test_epistemic_silver_concept_reads_and_casts_id(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    concept_file = Path(policy.bronze_data_path) / "sct2_Concept_Snapshot_123.txt"
    with open(concept_file, "w", encoding="utf-8") as f:
        f.write("id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n")
        f.write("138875005\t20020131\t1\t900000000000207008\t900000000000074008\n")

    intent = EpistemicSilverConceptIntent(policy=policy)

    # Act
    lazy_frame = intent.execute()
    df = lazy_frame.collect()

    # Assert
    assert len(df) == 1

    # Verify `id` is properly cast as a string type (Utf8/String in polars)
    assert df["id"].dtype == pl.String
    assert df["id"][0] == "138875005"


def test_epistemic_silver_concept_filters_inactive(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    concept_file = Path(policy.bronze_data_path) / "sct2_Concept_Snapshot_123.txt"
    with open(concept_file, "w", encoding="utf-8") as f:
        f.write("id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n")
        f.write("138875005\t20020131\t1\t900000000000207008\t900000000000074008\n")
        f.write("138875006\t20020131\t0\t900000000000207008\t900000000000074008\n")

    intent = EpistemicSilverConceptIntent(policy=policy)

    # Act
    df = intent.execute().collect()

    # Assert
    assert len(df) == 1
    assert "138875006" not in df["id"].to_list()


def test_epistemic_silver_concept_uuid_and_date(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    concept_file = Path(policy.bronze_data_path) / "sct2_Concept_Snapshot_123.txt"
    with open(concept_file, "w", encoding="utf-8") as f:
        f.write("id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n")
        f.write("138875005\t20020131\t1\t900000000000207008\t900000000000074008\n")

    intent = EpistemicSilverConceptIntent(policy=policy)

    # Act
    df = intent.execute().collect()

    # Assert
    import uuid
    from datetime import date

    expected_uuid = str(uuid.uuid5(uuid.UUID(policy.snomed_namespace_uuid), "138875005"))
    assert df["coreason_id"][0] == expected_uuid
    assert df["effectiveTime"][0] == date(2002, 1, 31)


def test_epistemic_silver_concept_missing_files(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    # Bronze dir is empty initially
    intent = EpistemicSilverConceptIntent(policy=policy)

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="No Concept Snapshot files found"):
        intent.execute()
