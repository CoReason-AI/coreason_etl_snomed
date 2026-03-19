# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

import sys
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

if sys.version_info < (3, 14):
    from dlt.pipeline.pipeline import Pipeline
else:
    Pipeline = MagicMock

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.load import EpistemicGoldDatabaseLoadIntent


@pytest.fixture
def mock_policy() -> EpistemicOntologyPolicy:
    """Provides a mocked EpistemicOntologyPolicy instance."""
    return EpistemicOntologyPolicy(
        umls_api_key="test_key",
        db_uri="postgresql://user:pass@localhost:5432/db",
        bronze_data_path="data/bronze/snomed/raw/",
    )


@pytest.fixture
def dummy_lazy_frames() -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Provides dummy LazyFrames for testing."""
    dim_concept = pl.LazyFrame({"coreason_id": ["1"], "name": ["Concept A"]})
    bridge_synonym = pl.LazyFrame({"coreason_id": ["1"], "synonyms": [["Synonym 1"]]})
    fact_relationship = pl.LazyFrame(
        {
            "coreason_id": ["rel1"],
            "source_coreason_id": ["1"],
            "destination_coreason_id": ["2"],
            "relationship_type_coreason_id": ["type1"],
        }
    )
    return dim_concept, bridge_synonym, fact_relationship


def test_epistemic_gold_database_load_intent_success(
    mock_policy: EpistemicOntologyPolicy,
    dummy_lazy_frames: tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame],
) -> None:
    """Test successful database loading using mocked dlt and polars."""
    dim_concept, bridge_synonym, fact_relationship = dummy_lazy_frames

    intent = EpistemicGoldDatabaseLoadIntent(policy=mock_policy)

    with (
        patch("coreason_etl_snomed.load.dlt.pipeline", return_value=MagicMock(spec=Pipeline)) as mock_pipeline_creator,
        patch.object(pl.LazyFrame, "sink_parquet") as mock_sink_parquet,
        patch("coreason_etl_snomed.load.tempfile.NamedTemporaryFile") as mock_tempfile,
        patch("coreason_etl_snomed.load.Path.unlink") as mock_unlink,
        patch("coreason_etl_snomed.load.Path.exists", return_value=True),
    ):
        mock_pipeline = mock_pipeline_creator.return_value
        # For mocking Pipeline properties cleanly
        mock_pipeline.run = MagicMock()
        mock_pipeline.run.return_value = "Mock Load Info"

        mock_temp_instance = MagicMock()
        mock_temp_instance.name = "mock_temp.parquet"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_instance

        # Execute load
        intent.execute(dim_concept, bridge_synonym, fact_relationship)

        # Verify pipeline created with proper config
        mock_pipeline_creator.assert_called_once()
        kwargs = mock_pipeline_creator.call_args[1]
        assert kwargs["pipeline_name"] == "snomed_ct_pipeline"
        assert kwargs["dataset_name"] == "gold"

        # Verify sink_parquet was called for each table
        assert mock_sink_parquet.call_count == 3

        # Verify dlt pipeline.run was called for each table with replace disposition
        assert mock_pipeline.run.call_count == 3

        # Verify table names were correctly passed to pipeline.run
        calls = mock_pipeline.run.call_args_list
        assert calls[0].kwargs["table_name"] == "snomed_gold_dim_concept"
        assert calls[1].kwargs["table_name"] == "snomed_gold_bridge_synonym"
        assert calls[2].kwargs["table_name"] == "snomed_gold_fact_relationship"

        # Verify temporary file was deleted for each table
        assert mock_unlink.call_count == 3


def test_epistemic_gold_database_load_intent_dlt_failure(
    mock_policy: EpistemicOntologyPolicy,
    dummy_lazy_frames: tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame],
) -> None:
    """Test behavior when dlt pipeline run fails."""
    dim_concept, bridge_synonym, fact_relationship = dummy_lazy_frames

    intent = EpistemicGoldDatabaseLoadIntent(policy=mock_policy)

    with (
        patch("coreason_etl_snomed.load.dlt.pipeline", return_value=MagicMock(spec=Pipeline)) as mock_pipeline_creator,
        patch.object(pl.LazyFrame, "sink_parquet"),
        patch("coreason_etl_snomed.load.tempfile.NamedTemporaryFile") as mock_tempfile,
        patch("coreason_etl_snomed.load.Path.unlink") as mock_unlink,
        patch("coreason_etl_snomed.load.Path.exists", return_value=True),
    ):
        mock_pipeline = mock_pipeline_creator.return_value
        mock_pipeline.run = MagicMock()
        mock_pipeline.run.side_effect = Exception("DLT Load Failed")

        mock_temp_instance = MagicMock()
        mock_temp_instance.name = "mock_temp.parquet"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_instance
        mock_tempfile.return_value.__exit__.return_value = False

        with pytest.raises(Exception, match="DLT Load Failed"):
            intent.execute(dim_concept, bridge_synonym, fact_relationship)

        # Verify cleanup still happens even on failure (for the table that failed)
        assert mock_unlink.call_count == 1


def test_epistemic_gold_database_load_intent_sink_parquet_failure(
    mock_policy: EpistemicOntologyPolicy,
    dummy_lazy_frames: tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame],
) -> None:
    """Test behavior when Polars sink_parquet fails."""
    dim_concept, bridge_synonym, fact_relationship = dummy_lazy_frames

    intent = EpistemicGoldDatabaseLoadIntent(policy=mock_policy)

    with (
        patch("coreason_etl_snomed.load.dlt.pipeline", return_value=MagicMock(spec=Pipeline)) as mock_pipeline_creator,
        patch.object(pl.LazyFrame, "sink_parquet", side_effect=Exception("Sink Parquet Failed")),
        patch("coreason_etl_snomed.load.tempfile.NamedTemporaryFile") as mock_tempfile,
        patch("coreason_etl_snomed.load.Path.unlink") as mock_unlink,
        patch("coreason_etl_snomed.load.Path.exists", return_value=True),
    ):
        mock_pipeline = mock_pipeline_creator.return_value
        mock_pipeline.run = MagicMock()

        mock_temp_instance = MagicMock()
        mock_temp_instance.name = "mock_temp.parquet"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_instance
        mock_tempfile.return_value.__exit__.return_value = False

        with pytest.raises(Exception, match="Sink Parquet Failed"):
            intent.execute(dim_concept, bridge_synonym, fact_relationship)

        # Verify pipeline.run was not called
        mock_pipeline.run.assert_not_called()

        # Verify cleanup still happens even if sink_parquet fails
        assert mock_unlink.call_count == 1
