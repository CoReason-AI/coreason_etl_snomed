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
from coreason_etl_snomed.main import EpistemicOntologyPipelineIntent


@pytest.fixture
def mock_policy() -> EpistemicOntologyPolicy:
    """Provides a mocked EpistemicOntologyPolicy instance."""
    return EpistemicOntologyPolicy(
        umls_api_key="test_key",
        db_uri="postgresql://user:pass@localhost:5432/db",
        bronze_data_path="data/bronze/snomed/raw/",
    )


def test_epistemic_ontology_pipeline_intent_success(mock_policy: EpistemicOntologyPolicy) -> None:
    """Test successful orchestration of the ETL pipeline."""
    intent = EpistemicOntologyPipelineIntent(policy=mock_policy)

    with (
        patch("coreason_etl_snomed.main.EpistemicOntologyFetchIntent.execute") as mock_fetch,
        patch("coreason_etl_snomed.main.EpistemicBronzeExtractionIntent.execute") as mock_extract,
        patch("coreason_etl_snomed.main.EpistemicSilverConceptIntent.execute") as mock_silver_concept,
        patch("coreason_etl_snomed.main.EpistemicSilverDescriptionIntent.execute") as mock_silver_description,
        patch("coreason_etl_snomed.main.EpistemicSilverRelationshipIntent.execute") as mock_silver_relationship,
        patch("coreason_etl_snomed.main.EpistemicGoldConceptIntent.execute") as mock_gold_concept,
        patch("coreason_etl_snomed.main.EpistemicGoldSynonymIntent.execute") as mock_gold_synonym,
        patch("coreason_etl_snomed.main.EpistemicGoldRelationshipIntent.execute") as mock_gold_relationship,
        patch("coreason_etl_snomed.main.EpistemicGoldDatabaseLoadIntent.execute") as mock_load,
    ):
        mock_archive_path = MagicMock(spec=Path)
        mock_fetch.return_value = mock_archive_path

        mock_silver_concept_lf = MagicMock(spec=pl.LazyFrame)
        mock_silver_description_lf = MagicMock(spec=pl.LazyFrame)
        mock_silver_relationship_lf = MagicMock(spec=pl.LazyFrame)
        mock_silver_concept.return_value = mock_silver_concept_lf
        mock_silver_description.return_value = mock_silver_description_lf
        mock_silver_relationship.return_value = mock_silver_relationship_lf

        mock_gold_concept_lf = MagicMock(spec=pl.LazyFrame)
        mock_gold_synonym_lf = MagicMock(spec=pl.LazyFrame)
        mock_gold_relationship_lf = MagicMock(spec=pl.LazyFrame)
        mock_gold_concept.return_value = mock_gold_concept_lf
        mock_gold_synonym.return_value = mock_gold_synonym_lf
        mock_gold_relationship.return_value = mock_gold_relationship_lf

        # Execute the pipeline
        intent.execute()

        # Assert correct methods were called
        mock_fetch.assert_called_once()
        mock_extract.assert_called_once()
        mock_silver_concept.assert_called_once()
        mock_silver_description.assert_called_once()
        mock_silver_relationship.assert_called_once()

        mock_gold_concept.assert_called_once_with(
            silver_concept=mock_silver_concept_lf, silver_description=mock_silver_description_lf
        )
        mock_gold_synonym.assert_called_once_with(silver_description=mock_silver_description_lf)
        mock_gold_relationship.assert_called_once_with(silver_relationship=mock_silver_relationship_lf)

        mock_load.assert_called_once_with(
            dim_concept=mock_gold_concept_lf,
            bridge_synonym=mock_gold_synonym_lf,
            fact_relationship=mock_gold_relationship_lf,
        )


def test_epistemic_ontology_pipeline_intent_failure(mock_policy: EpistemicOntologyPolicy) -> None:
    """Test the pipeline propagates exceptions when a step fails."""
    intent = EpistemicOntologyPipelineIntent(policy=mock_policy)

    with (
        patch("coreason_etl_snomed.main.EpistemicOntologyFetchIntent.execute", side_effect=Exception("Fetch Failed")),
        patch("coreason_etl_snomed.main.EpistemicBronzeExtractionIntent.execute") as mock_extract,
    ):
        with pytest.raises(Exception, match="Fetch Failed"):
            intent.execute()

        mock_extract.assert_not_called()


def test_main_execution() -> None:
    """Test the main entry point execution."""
    with (
        patch("coreason_etl_snomed.main.EpistemicOntologyPolicy") as mock_policy_cls,
        patch("coreason_etl_snomed.main.EpistemicOntologyPipelineIntent") as mock_intent_cls,
    ):
        from coreason_etl_snomed.main import main

        mock_policy_instance = mock_policy_cls.return_value
        mock_intent_instance = mock_intent_cls.return_value

        main()

        mock_policy_cls.assert_called_once()
        mock_intent_cls.assert_called_once_with(policy=mock_policy_instance)
        mock_intent_instance.execute.assert_called_once()
