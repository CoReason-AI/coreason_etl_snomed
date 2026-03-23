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
import tempfile
from pathlib import Path
from typing import Any

if sys.version_info < (3, 14):  # pragma: no cover
    import dlt
else:

    class _MockDlt:
        def pipeline(self, *args: Any, **kwargs: Any) -> Any: ...

    dlt = _MockDlt()


import polars as pl
from pydantic import BaseModel, Field

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.utils.logger import logger


class BaseDatabaseLoadIntent(BaseModel):
    """
    AGENT INSTRUCTION: Base class providing shared loading infrastructure utilizing dlt and temporary
    parquet files to load datasets while preventing OOM conditions.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def _get_pipeline(self, dataset_name: str) -> Any:
        import sys

        if sys.version_info < (3, 14):  # pragma: no cover
            from dlt.destinations import postgres

            pipeline_obj = dlt.pipeline(
                pipeline_name="snomed_ct_pipeline",
                destination=postgres(credentials=self.policy.db_uri.get_secret_value()),
                dataset_name=dataset_name,
            )
        else:
            pipeline_obj = dlt.pipeline(
                pipeline_name="snomed_ct_pipeline",
                dataset_name=dataset_name,
            )
        return pipeline_obj

    def _load_table(self, pipeline: Any, lf: pl.LazyFrame, table_name: str) -> None:
        """
        Helper method to stream a LazyFrame to a temporary parquet file and load it via dlt.

        Args:
            pipeline (Any): The initialized dlt pipeline.
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


class EpistemicBronzeDatabaseLoadIntent(BaseDatabaseLoadIntent):
    """
    AGENT INSTRUCTION: This class is responsible for loading the raw Bronze representation of the
    ontology (unfiltered text snapshot files) directly into the CoReason PostgreSQL Knowledge Graph.
    """

    def execute(self) -> None:
        """
        Executes the bronze database load intent by dynamically scanning the local extracted
        Bronze text files, casting all columns to String to avoid schema mismatch,
        and loading them to the database.
        """
        logger.info("Starting Bronze database load.")

        bronze_dir = Path(self.policy.bronze_data_path)
        if not bronze_dir.exists():
            error_msg = f"Bronze directory {bronze_dir} does not exist."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        pipeline = self._get_pipeline(dataset_name="bronze")

        try:
            self._load_bronze_file(bronze_dir, "sct2_Concept_Snapshot_*.txt", pipeline, "snomed_bronze_concept")
            self._load_bronze_file(bronze_dir, "sct2_Description_Snapshot_*.txt", pipeline, "snomed_bronze_description")
            self._load_bronze_file(
                bronze_dir, "sct2_Relationship_Snapshot_*.txt", pipeline, "snomed_bronze_relationship"
            )
            logger.info("Successfully completed Bronze database load.")
        except Exception as e:
            logger.exception("Failed to load Bronze database tables.")
            raise e

    def _load_bronze_file(self, bronze_dir: Path, glob_pattern: str, pipeline: Any, table_name: str) -> None:
        """Helper to scan raw text files, enforce String schema, and load via DLT."""
        files = list(bronze_dir.glob(glob_pattern))
        if not files:
            logger.warning(f"No Bronze files found for pattern {glob_pattern}")
            return

        file_paths = [str(f) for f in files]
        logger.info(f"Scanning Bronze files for {table_name}: {file_paths}")

        # Scan csv but infer all columns as String to prevent arbitrary parsing errors in raw data
        lf = pl.scan_csv(file_paths, separator="\t", infer_schema_length=0)
        self._load_table(pipeline, lf, table_name)


class EpistemicSilverDatabaseLoadIntent(BaseDatabaseLoadIntent):
    """
    AGENT INSTRUCTION: This class is responsible for loading the Silver representation of the
    ontology (cleaned, typed, and filtered snapshots) into the CoReason PostgreSQL Knowledge Graph.
    """

    def execute(
        self, silver_concept: pl.LazyFrame, silver_description: pl.LazyFrame, silver_relationship: pl.LazyFrame
    ) -> None:
        """
        Executes the silver database load intent.

        Args:
            silver_concept (pl.LazyFrame): The silver concept table.
            silver_description (pl.LazyFrame): The silver description table.
            silver_relationship (pl.LazyFrame): The silver relationship table.
        """
        logger.info("Starting Silver database load.")

        pipeline = self._get_pipeline(dataset_name="silver")

        try:
            self._load_table(pipeline, silver_concept, "snomed_silver_concept")
            self._load_table(pipeline, silver_description, "snomed_silver_description")
            self._load_table(pipeline, silver_relationship, "snomed_silver_relationship")
            logger.info("Successfully completed Silver database load.")
        except Exception as e:
            logger.exception("Failed to load Silver database tables.")
            raise e


class EpistemicGoldDatabaseLoadIntent(BaseDatabaseLoadIntent):
    """
    AGENT INSTRUCTION: This class is responsible for loading the final Gold representation of the
    ontology (Concepts, Synonyms, and Relationships) into the CoReason PostgreSQL Knowledge Graph.
    """

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

        pipeline = self._get_pipeline(dataset_name="gold")

        try:
            self._load_table(pipeline, dim_concept, "snomed_gold_dim_concept")
            self._load_table(pipeline, bridge_synonym, "snomed_gold_bridge_synonym")
            self._load_table(pipeline, fact_relationship, "snomed_gold_fact_relationship")
            logger.info("Successfully completed Gold database load.")
        except Exception as e:
            logger.exception("Failed to load Gold database tables.")
            raise e
