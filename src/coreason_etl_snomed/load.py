# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

import tempfile
from pathlib import Path

import dlt
import polars as pl
from pydantic import BaseModel, Field

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.utils.logger import logger


class EpistemicGoldDatabaseLoadIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for loading the final Gold representation of the
    ontology (Concepts, Synonyms, and Relationships) into the CoReason PostgreSQL Knowledge Graph.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self, dim_concept: pl.LazyFrame, bridge_synonym: pl.LazyFrame, fact_relationship: pl.LazyFrame) -> None:
        """
        Executes the database load intent. To avoid OOM errors, it uses Polars' sink_parquet
        to write the data to temporary parquet files, and then uses dlt to bulk-load the files
        into the PostgreSQL database using a replace write disposition.

        Args:
            dim_concept (pl.LazyFrame): The gold dimension table for concepts.
            bridge_synonym (pl.LazyFrame): The gold bridge table for concept synonyms.
            fact_relationship (pl.LazyFrame): The gold fact table for concept relationships.
        """
        logger.info("Starting Gold database load.")

        from dlt.destinations import postgres
        from dlt.pipeline.pipeline import Pipeline

        pipeline: Pipeline = dlt.pipeline(
            pipeline_name="snomed_ct_pipeline",
            destination=postgres(credentials=self.policy.db_uri.get_secret_value()),
            dataset_name="ontology",
        )

        try:
            self._load_table(pipeline, dim_concept, "dim_snomed_concept")
            self._load_table(pipeline, bridge_synonym, "bridge_snomed_synonym")
            self._load_table(pipeline, fact_relationship, "fact_snomed_relationship")
            logger.info("Successfully completed Gold database load.")
        except Exception as e:
            logger.exception("Failed to load Gold database tables.")
            raise e

    def _load_table(self, pipeline: "dlt.Pipeline", lf: pl.LazyFrame, table_name: str) -> None:
        """
        Helper method to stream a LazyFrame to a temporary parquet file and load it via dlt.

        Args:
            pipeline (dlt.Pipeline): The initialized dlt pipeline.
            lf (pl.LazyFrame): The lazy evaluated dataframe to load.
            table_name (str): The name of the target database table.
        """
        logger.info(f"Loading table {table_name}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
                temp_file_path = Path(temp_file.name)

            # To avoid OOM, stream output to parquet
            lf.sink_parquet(temp_file_path)

            logger.info(f"Successfully materialized {table_name} to {temp_file_path}, loading into dlt.")

            # Load the generated parquet file via dlt into PostgreSQL
            # We use `write_disposition="replace"` to ensure the database matches the current Snapshot
            load_info = pipeline.run(
                [str(temp_file_path)],
                table_name=table_name,
                write_disposition="replace",
                loader_file_format="parquet",
            )
            logger.info(f"Loaded {table_name}: {load_info}")

        except Exception as e:
            logger.exception(f"Failed to load table {table_name}")
            raise e
        finally:
            if "temp_file_path" in locals() and temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
                logger.info(f"Cleaned up temporary file {temp_file_path}")
