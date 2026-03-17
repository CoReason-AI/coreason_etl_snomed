# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

from pydantic import BaseModel, Field

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.extract import EpistemicBronzeExtractionIntent
from coreason_etl_snomed.fetch import EpistemicOntologyFetchIntent
from coreason_etl_snomed.load import EpistemicGoldDatabaseLoadIntent
from coreason_etl_snomed.transform import (
    EpistemicGoldConceptIntent,
    EpistemicGoldRelationshipIntent,
    EpistemicGoldSynonymIntent,
    EpistemicSilverConceptIntent,
    EpistemicSilverDescriptionIntent,
    EpistemicSilverRelationshipIntent,
)
from coreason_etl_snomed.utils.logger import logger


class EpistemicOntologyPipelineIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class orchestrates the complete SNOMED CT Medallion ETL pipeline.
    It executes the Fetch, Extract, Transform (Silver and Gold), and Load intents sequentially.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self) -> None:
        """
        Executes the end-to-end Medallion ETL pipeline for SNOMED CT data.
        """
        logger.info("Starting Epistemic Ontology Pipeline execution.")

        try:
            # 1. Fetch
            fetch_intent = EpistemicOntologyFetchIntent(policy=self.policy)
            archive_path = fetch_intent.execute()

            # 2. Extract (Bronze)
            extract_intent = EpistemicBronzeExtractionIntent(policy=self.policy, archive_path=archive_path)
            extract_intent.execute()

            # 3. Transform (Silver)
            silver_concept_intent = EpistemicSilverConceptIntent(policy=self.policy)
            silver_concept_lf = silver_concept_intent.execute()

            silver_description_intent = EpistemicSilverDescriptionIntent(policy=self.policy)
            silver_description_lf = silver_description_intent.execute()

            silver_relationship_intent = EpistemicSilverRelationshipIntent(policy=self.policy)
            silver_relationship_lf = silver_relationship_intent.execute()

            # 4. Transform (Gold)
            gold_concept_intent = EpistemicGoldConceptIntent(policy=self.policy)
            gold_concept_lf = gold_concept_intent.execute(
                silver_concept=silver_concept_lf, silver_description=silver_description_lf
            )

            gold_synonym_intent = EpistemicGoldSynonymIntent(policy=self.policy)
            gold_synonym_lf = gold_synonym_intent.execute(silver_description=silver_description_lf)

            gold_relationship_intent = EpistemicGoldRelationshipIntent(policy=self.policy)
            gold_relationship_lf = gold_relationship_intent.execute(silver_relationship=silver_relationship_lf)

            # 5. Load
            load_intent = EpistemicGoldDatabaseLoadIntent(policy=self.policy)
            load_intent.execute(
                dim_concept=gold_concept_lf, bridge_synonym=gold_synonym_lf, fact_relationship=gold_relationship_lf
            )

            logger.info("Successfully completed Epistemic Ontology Pipeline execution.")

        except Exception as e:
            logger.exception("Pipeline execution failed.")
            raise e


def main() -> None:
    """
    AGENT INSTRUCTION: Entry point for the Epistemic Ontology Pipeline execution.
    It instantiates the policy and executes the pipeline intent.
    """
    logger.info("Initializing Epistemic Ontology Policy.")
    policy = EpistemicOntologyPolicy()
    intent = EpistemicOntologyPipelineIntent(policy=policy)
    intent.execute()


if __name__ == "__main__":  # pragma: no cover
    main()
