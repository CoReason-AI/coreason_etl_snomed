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
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.load import (
    EpistemicBronzeDatabaseLoadIntent,
    EpistemicGoldDatabaseLoadIntent,
    EpistemicSilverDatabaseLoadIntent,
)


@pytest.fixture
def mock_policy() -> EpistemicOntologyPolicy:
    """Provides a mocked EpistemicOntologyPolicy instance."""
    return EpistemicOntologyPolicy(
        umls_api_key="test_key",
        db_uri="postgresql://user:pass@localhost:5432/db",
        bronze_data_path="data/bronze/snomed/raw/",
    )


@pytest.fixture
def dummy_lazy_frames() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Provides dummy LazyFrames for testing."""
    dim_concept = pl.LazyFrame({"coreason_id": ["1"], "name": ["Concept A"]})
    fact_relationship = pl.LazyFrame(
        {
            "coreason_id": ["rel1"],
            "source_coreason_id": ["1"],
            "destination_coreason_id": ["2"],
            "relationship_type_coreason_id": ["type1"],
        }
    )
    return dim_concept, fact_relationship


def test_epistemic_gold_database_load_intent_success(
    mock_policy: EpistemicOntologyPolicy,
    dummy_lazy_frames: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    """Test successful database loading using mocked polars write_database."""
    dim_concept, fact_relationship = dummy_lazy_frames
    intent = EpistemicGoldDatabaseLoadIntent(policy=mock_policy)

    with patch("polars.DataFrame.write_database") as mock_write_db:
        intent.execute(dim_concept, fact_relationship)
        
        # Verify write_database was called exactly twice (for the 2 gold tables)
        assert mock_write_db.call_count == 2
        
        # Verify the arguments passed to the last write_database call
        kwargs = mock_write_db.call_args[1]
        assert kwargs["if_table_exists"] == "replace"
        assert kwargs["engine"] == "adbc"


def test_epistemic_bronze_database_load_intent_success(mock_policy: EpistemicOntologyPolicy, tmp_path: Path) -> None:
    """Test successful Bronze database loading."""
    mock_policy.bronze_data_path = str(tmp_path)

    # Create dummy bronze files
    (tmp_path / "CONCEPT.csv").write_text("dummy\n1\n")
    (tmp_path / "CONCEPT_SYNONYM.csv").write_text("dummy\n1\n")
    (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text("dummy\n1\n")

    intent = EpistemicBronzeDatabaseLoadIntent(policy=mock_policy)

    with patch("polars.DataFrame.write_database") as mock_write_db:
        intent.execute()
        # Verify write_database was called exactly 3 times (for 3 bronze tables)
        assert mock_write_db.call_count == 3


def test_epistemic_bronze_database_load_intent_no_dir(mock_policy: EpistemicOntologyPolicy) -> None:
    """Test behavior when Bronze directory doesn't exist."""
    intent = EpistemicBronzeDatabaseLoadIntent(policy=mock_policy)

    with pytest.raises(FileNotFoundError, match=r"Bronze directory .* does not exist"):
        intent.execute()


def test_epistemic_bronze_database_load_intent_no_files(mock_policy: EpistemicOntologyPolicy, tmp_path: Path) -> None:
    """Test behavior when Bronze directory is empty."""
    mock_policy.bronze_data_path = str(tmp_path)
    intent = EpistemicBronzeDatabaseLoadIntent(policy=mock_policy)

    with patch("coreason_etl_snomed.load.EpistemicBronzeDatabaseLoadIntent._load_table") as mock_load_table:
        intent.execute()
        # Should log warning but not throw error and not attempt to load anything
        mock_load_table.assert_not_called()


def test_epistemic_silver_database_load_intent_success(
    mock_policy: EpistemicOntologyPolicy,
    dummy_lazy_frames: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    """Test successful Silver database loading."""
    dim_concept, fact_relationship = dummy_lazy_frames
    dummy_silver_desc = pl.LazyFrame({"coreason_id": ["1"], "term": ["desc"]})

    intent = EpistemicSilverDatabaseLoadIntent(policy=mock_policy)

    with patch("polars.DataFrame.write_database") as mock_write_db:
        intent.execute(dim_concept, dummy_silver_desc, fact_relationship)
        # Silver loads 3 tables
        assert mock_write_db.call_count == 3


def test_database_load_intent_db_failure(
    mock_policy: EpistemicOntologyPolicy,
    dummy_lazy_frames: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    """Test behavior when the database load throws an exception."""
    dim_concept, fact_relationship = dummy_lazy_frames
    intent = EpistemicGoldDatabaseLoadIntent(policy=mock_policy)

    with patch("polars.DataFrame.write_database", side_effect=Exception("DB Connection Failed")):
        with pytest.raises(Exception, match="DB Connection Failed"):
            intent.execute(dim_concept, fact_relationship)
