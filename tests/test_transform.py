# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

import uuid
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.transform import (
    NAMESPACE_SNOMED,
    EpistemicGoldConceptIntent,
    EpistemicGoldRelationshipIntent,
    EpistemicGoldSynonymIntent,
    EpistemicSilverConceptIntent,
    EpistemicSilverDescriptionIntent,
    EpistemicSilverRelationshipIntent,
)


@pytest.fixture
def policy(tmp_path: Path) -> EpistemicOntologyPolicy:
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    return EpistemicOntologyPolicy(
        umls_api_key="fake",
        db_uri="postgresql://user:pass@localhost:5432/db",
        bronze_data_path=str(bronze_dir),
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

    # Verify `id` is properly cast as a string type (Utf8/String in polars) and aliased as `source_id`
    assert df["source_id"].dtype == pl.String
    assert df["source_id"][0] == "138875005"


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
    assert "138875006" not in df["source_id"].to_list()


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
    expected_uuid = str(uuid.uuid5(NAMESPACE_SNOMED, "138875005"))
    assert df["coreason_id"][0] == expected_uuid
    assert df["effectiveTime"][0] == date(2002, 1, 31)


def test_epistemic_silver_concept_missing_files(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    # Bronze dir is empty initially
    intent = EpistemicSilverConceptIntent(policy=policy)

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="No Concept Snapshot files found"):
        intent.execute()


def test_epistemic_silver_description_reads_and_casts_id(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    desc_file = Path(policy.bronze_data_path) / "sct2_Description_Snapshot_123.txt"
    with open(desc_file, "w", encoding="utf-8") as f:
        f.write("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n")
        f.write(
            "11111\t20020131\t1\t900000000000207008\t138875005\t"
            "en\t900000000000003001\tSNOMED CT Concept\t900000000000020002\n"
        )

    intent = EpistemicSilverDescriptionIntent(policy=policy)

    # Act
    lazy_frame = intent.execute()
    df = lazy_frame.collect()

    # Assert
    assert len(df) == 1

    # Verify `id` and `conceptId` are properly cast as string type
    assert df["source_id"].dtype == pl.String
    assert df["source_id"][0] == "11111"
    assert df["conceptId"].dtype == pl.String
    assert df["conceptId"][0] == "138875005"


def test_epistemic_silver_description_filters_inactive(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    desc_file = Path(policy.bronze_data_path) / "sct2_Description_Snapshot_123.txt"
    with open(desc_file, "w", encoding="utf-8") as f:
        f.write("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n")
        f.write(
            "11111\t20020131\t1\t900000000000207008\t138875005\t"
            "en\t900000000000003001\tSNOMED CT Concept\t900000000000020002\n"
        )
        f.write(
            "22222\t20020131\t0\t900000000000207008\t138875006\t"
            "en\t900000000000013009\tAnother Concept\t900000000000020002\n"
        )

    intent = EpistemicSilverDescriptionIntent(policy=policy)

    # Act
    df = intent.execute().collect()

    # Assert
    assert len(df) == 1
    assert "22222" not in df["source_id"].to_list()


def test_epistemic_silver_description_uuid_and_date(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    desc_file = Path(policy.bronze_data_path) / "sct2_Description_Snapshot_123.txt"
    with open(desc_file, "w", encoding="utf-8") as f:
        f.write("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n")
        f.write(
            "11111\t20020131\t1\t900000000000207008\t138875005\t"
            "en\t900000000000003001\tSNOMED CT Concept\t900000000000020002\n"
        )

    intent = EpistemicSilverDescriptionIntent(policy=policy)

    # Act
    df = intent.execute().collect()

    # Assert
    expected_desc_uuid = str(uuid.uuid5(NAMESPACE_SNOMED, "11111"))
    expected_concept_uuid = str(uuid.uuid5(NAMESPACE_SNOMED, "138875005"))

    assert df["coreason_id"][0] == expected_desc_uuid
    assert df["concept_coreason_id"][0] == expected_concept_uuid
    assert df["effectiveTime"][0] == date(2002, 1, 31)


def test_epistemic_silver_description_missing_files(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    # Bronze dir is empty initially
    intent = EpistemicSilverDescriptionIntent(policy=policy)

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="No Description Snapshot files found"):
        intent.execute()


def test_epistemic_silver_relationship_reads_and_casts_id(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    rel_file = Path(policy.bronze_data_path) / "sct2_Relationship_Snapshot_123.txt"
    with open(rel_file, "w", encoding="utf-8") as f:
        f.write(
            "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n"
        )
        f.write(
            "11111\t20020131\t1\t900000000000207008\t138875005\t"
            "138875006\t0\t116680003\t900000000000010007\t900000000000214000\n"
        )

    intent = EpistemicSilverRelationshipIntent(policy=policy)

    # Act
    lazy_frame = intent.execute()
    df = lazy_frame.collect()

    # Assert
    assert len(df) == 1

    # Verify ID fields are properly cast as string type
    assert df["source_id"].dtype == pl.String
    assert df["source_id"][0] == "11111"
    assert df["sourceId"].dtype == pl.String
    assert df["sourceId"][0] == "138875005"
    assert df["destinationId"].dtype == pl.String
    assert df["destinationId"][0] == "138875006"
    assert df["typeId"].dtype == pl.String
    assert df["typeId"][0] == "116680003"


def test_epistemic_silver_relationship_filters_inactive(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    rel_file = Path(policy.bronze_data_path) / "sct2_Relationship_Snapshot_123.txt"
    with open(rel_file, "w", encoding="utf-8") as f:
        f.write(
            "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n"
        )
        f.write(
            "11111\t20020131\t1\t900000000000207008\t138875005\t"
            "138875006\t0\t116680003\t900000000000010007\t900000000000214000\n"
        )
        f.write(
            "22222\t20020131\t0\t900000000000207008\t138875005\t"
            "138875007\t0\t116680003\t900000000000010007\t900000000000214000\n"
        )

    intent = EpistemicSilverRelationshipIntent(policy=policy)

    # Act
    df = intent.execute().collect()

    # Assert
    assert len(df) == 1
    assert "22222" not in df["source_id"].to_list()


def test_epistemic_silver_relationship_uuid_and_date(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    rel_file = Path(policy.bronze_data_path) / "sct2_Relationship_Snapshot_123.txt"
    with open(rel_file, "w", encoding="utf-8") as f:
        f.write(
            "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n"
        )
        f.write(
            "11111\t20020131\t1\t900000000000207008\t138875005\t"
            "138875006\t0\t116680003\t900000000000010007\t900000000000214000\n"
        )

    intent = EpistemicSilverRelationshipIntent(policy=policy)

    # Act
    df = intent.execute().collect()

    # Assert
    ns = NAMESPACE_SNOMED
    expected_rel_uuid = str(uuid.uuid5(ns, "11111"))
    expected_source_uuid = str(uuid.uuid5(ns, "138875005"))
    expected_dest_uuid = str(uuid.uuid5(ns, "138875006"))
    expected_type_uuid = str(uuid.uuid5(ns, "116680003"))

    assert df["coreason_id"][0] == expected_rel_uuid
    assert df["source_coreason_id"][0] == expected_source_uuid
    assert df["destination_coreason_id"][0] == expected_dest_uuid
    assert df["type_coreason_id"][0] == expected_type_uuid
    assert df["effectiveTime"][0] == date(2002, 1, 31)


def test_epistemic_silver_relationship_missing_files(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    # Bronze dir is empty initially
    intent = EpistemicSilverRelationshipIntent(policy=policy)

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="No Relationship Snapshot files found"):
        intent.execute()


def test_epistemic_gold_concept_joins_name(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    concept_lf = pl.LazyFrame(
        {
            "coreason_id": ["concept-1", "concept-2"],
            "id": ["1", "2"],
        }
    )

    desc_lf = pl.LazyFrame(
        {
            "concept_coreason_id": ["concept-1", "concept-1", "concept-2"],
            "typeId": ["900000000000003001", "900000000000013009", "900000000000003001"],
            "active": [1, 1, 0],  # active Fully Specified, active Synonym, inactive Fully Specified
            "term": ["Name 1", "Synonym 1", "Name 2"],
        }
    )

    intent = EpistemicGoldConceptIntent(policy=policy)

    # Act
    df = intent.execute(concept_lf, desc_lf).collect()

    # Assert
    assert len(df) == 2

    row1 = df.filter(pl.col("coreason_id") == "concept-1").row(0, named=True)
    assert row1["name"] == "Name 1"

    row2 = df.filter(pl.col("coreason_id") == "concept-2").row(0, named=True)
    assert row2["name"] is None  # Since the FSN is inactive, left join leaves name as null


def test_epistemic_gold_concept_missing_description(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    concept_lf = pl.LazyFrame(
        {
            "coreason_id": ["concept-1"],
            "id": ["1"],
        }
    )

    desc_lf = pl.LazyFrame(
        {"concept_coreason_id": [], "typeId": [], "active": [], "term": []},
        schema={"concept_coreason_id": pl.String, "typeId": pl.String, "active": pl.Int64, "term": pl.String},
    )

    intent = EpistemicGoldConceptIntent(policy=policy)

    # Act
    df = intent.execute(concept_lf, desc_lf).collect()

    # Assert
    assert len(df) == 1

    row1 = df.filter(pl.col("coreason_id") == "concept-1").row(0, named=True)
    assert row1["name"] is None


def test_epistemic_gold_synonym_filters_and_aggregates(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    desc_lf = pl.LazyFrame(
        {
            "concept_coreason_id": ["concept-1", "concept-1", "concept-2", "concept-3", "concept-3"],
            "typeId": [
                "900000000000013009",  # Valid synonym
                "900000000000013009",  # Valid synonym
                "900000000000003001",  # Invalid type (Fully Specified Name)
                "900000000000013009",  # Inactive synonym
                "900000000000013009",  # Valid synonym
            ],
            "active": [1, 1, 1, 0, 1],
            "term": ["Synonym A", "Synonym B", "Not a synonym", "Inactive synonym", "Synonym C"],
        }
    )

    intent = EpistemicGoldSynonymIntent(policy=policy)

    # Act
    df = intent.execute(desc_lf).collect()

    # Assert
    assert len(df) == 2  # concept-1 and concept-3 should have synonyms

    # concept-1
    row1 = df.filter(pl.col("coreason_id") == "concept-1").row(0, named=True)
    assert sorted(row1["synonyms"]) == ["Synonym A", "Synonym B"]

    # concept-3
    row3 = df.filter(pl.col("coreason_id") == "concept-3").row(0, named=True)
    assert row3["synonyms"] == ["Synonym C"]

    # concept-2 should not exist because it had no valid synonyms
    assert len(df.filter(pl.col("coreason_id") == "concept-2")) == 0


def test_epistemic_gold_synonym_empty_description(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    desc_lf = pl.LazyFrame(
        {"concept_coreason_id": [], "typeId": [], "active": [], "term": []},
        schema={"concept_coreason_id": pl.String, "typeId": pl.String, "active": pl.Int64, "term": pl.String},
    )

    intent = EpistemicGoldSynonymIntent(policy=policy)

    # Act
    df = intent.execute(desc_lf).collect()

    # Assert
    assert len(df) == 0
    assert "coreason_id" in df.columns
    assert "synonyms" in df.columns


def test_epistemic_gold_relationship_shapes_schema(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    silver_rel_lf = pl.LazyFrame(
        {
            "coreason_id": ["rel-1", "rel-2"],
            "effectiveTime": [date(2002, 1, 31), date(2002, 1, 31)],
            "active": [1, 1],
            "moduleId": ["mod-1", "mod-2"],
            "sourceId": ["src-1", "src-2"],
            "destinationId": ["dest-1", "dest-2"],
            "relationshipGroup": ["0", "1"],
            "typeId": ["type-1", "type-2"],
            "characteristicTypeId": ["char-1", "char-2"],
            "modifierId": ["mod-1", "mod-2"],
            "source_coreason_id": ["src-core-1", "src-core-2"],
            "destination_coreason_id": ["dest-core-1", "dest-core-2"],
            "type_coreason_id": ["type-core-1", "type-core-2"],
        }
    )

    intent = EpistemicGoldRelationshipIntent(policy=policy)

    # Act
    df = intent.execute(silver_rel_lf).collect()

    # Assert
    assert len(df) == 2
    assert df.columns == [
        "coreason_id",
        "source_coreason_id",
        "destination_coreason_id",
        "relationshipGroup",
        "characteristicTypeId",
        "modifierId",
        "relationship_type_coreason_id",
    ]

    row1 = df.row(0, named=True)
    assert row1["coreason_id"] == "rel-1"
    assert row1["source_coreason_id"] == "src-core-1"
    assert row1["destination_coreason_id"] == "dest-core-1"
    assert row1["relationshipGroup"] == "0"
    assert row1["characteristicTypeId"] == "char-1"
    assert row1["modifierId"] == "mod-1"
    assert row1["relationship_type_coreason_id"] == "type-core-1"


def test_epistemic_gold_relationship_empty_data(policy: EpistemicOntologyPolicy) -> None:
    # Arrange
    silver_rel_lf = pl.LazyFrame(
        {
            "coreason_id": [],
            "source_coreason_id": [],
            "destination_coreason_id": [],
            "relationshipGroup": [],
            "characteristicTypeId": [],
            "modifierId": [],
            "type_coreason_id": [],
        },
        schema={
            "coreason_id": pl.String,
            "source_coreason_id": pl.String,
            "destination_coreason_id": pl.String,
            "relationshipGroup": pl.String,
            "characteristicTypeId": pl.String,
            "modifierId": pl.String,
            "type_coreason_id": pl.String,
        },
    )

    intent = EpistemicGoldRelationshipIntent(policy=policy)

    # Act
    df = intent.execute(silver_rel_lf).collect()

    # Assert
    assert len(df) == 0
    assert df.columns == [
        "coreason_id",
        "source_coreason_id",
        "destination_coreason_id",
        "relationshipGroup",
        "characteristicTypeId",
        "modifierId",
        "relationship_type_coreason_id",
    ]
