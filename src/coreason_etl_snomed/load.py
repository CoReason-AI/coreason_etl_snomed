# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

import polars as pl
from pathlib import Path
from pydantic import BaseModel, Field

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.utils.logger import logger


class BaseDatabaseLoadIntent(BaseModel):
    """
    AGENT INSTRUCTION: Base class providing shared loading infrastructure utilizing 
    native Polars ADBC to load datasets directly to PostgreSQL.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def _load_table(self, lf: pl.LazyFrame, table_name: str) -> None:
        """
        Helper method to collect a LazyFrame and write it directly to Postgres.
        """
        logger.info(f"Loading table {table_name}")

        try:
            # Collect the lazy frame into memory
            df = lf.collect()
            
            # Write directly to PostgreSQL using ADBC
            df.write_database(
                table_name=table_name,
                connection=self.policy.db_uri.get_secret_value(),
                if_table_exists="replace",
                engine="adbc"
            )
            
            logger.info(f"Loaded {table_name} successfully.")
        except Exception as e:
            logger.exception(f"Failed to load table {table_name}")
            raise e


class EpistemicBronzeDatabaseLoadIntent(BaseDatabaseLoadIntent):
    """
    AGENT INSTRUCTION: This class is responsible for loading the raw Bronze representation 
    of the ontology (OMOP CSV files) directly into the CoReason PostgreSQL Knowledge Graph.
    """

    def execute(self) -> None:
        logger.info("Starting Bronze database load.")

        bronze_dir = Path(self.policy.bronze_data_path)
        if not bronze_dir.exists():
            error_msg = f"Bronze directory {bronze_dir} does not exist."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self._load_bronze_file(bronze_dir, "CONCEPT.csv", "snomed_bronze_concept")
            self._load_bronze_file(bronze_dir, "CONCEPT_SYNONYM.csv", "snomed_bronze_description")
            self._load_bronze_file(bronze_dir, "CONCEPT_RELATIONSHIP.csv", "snomed_bronze_relationship")
            logger.info("Successfully completed Bronze database load.")
        except Exception as e:
            logger.exception("Failed to load Bronze database tables.")
            raise e

    def _load_bronze_file(self, bronze_dir: Path, filename: str, table_name: str) -> None:
        """Helper to scan raw text files, enforce String schema, and load via Polars."""
        file_path = bronze_dir / filename
        if not file_path.exists():
            logger.warning(f"No Bronze file found for {filename}")
            return

        logger.info(f"Scanning Bronze file for {table_name}: {file_path}")

        # Scan csv, disable quote_char for OMOP tabs, infer all columns as String 
        lf = pl.scan_csv(file_path, separator="\t", infer_schema_length=0, quote_char=None, ignore_errors=True)
        self._load_table(lf, table_name)


class EpistemicSilverDatabaseLoadIntent(BaseDatabaseLoadIntent):
    """
    AGENT INSTRUCTION: This class is responsible for loading the Silver representation.
    """

    def execute(
        self, silver_concept: pl.LazyFrame, silver_description: pl.LazyFrame, silver_relationship: pl.LazyFrame
    ) -> None:
        logger.info("Starting Silver database load.")
        try:
            self._load_table(silver_concept, "snomed_silver_concept")
            self._load_table(silver_description, "snomed_silver_description")
            self._load_table(silver_relationship, "snomed_silver_relationship")
            logger.info("Successfully completed Silver database load.")
        except Exception as e:
            logger.exception("Failed to load Silver database tables.")
            raise e


class EpistemicGoldDatabaseLoadIntent(BaseDatabaseLoadIntent):
    """
    AGENT INSTRUCTION: This class is responsible for loading the final Gold representation.
    """

    def execute(self, dim_concept: pl.LazyFrame, fact_relationship: pl.LazyFrame) -> None:
        logger.info("Starting Gold database load.")
        try:
            self._load_table(dim_concept, "snomed_gold_dim_concept")
            # Removed self._load_table(bridge_synonym, ...)
            self._load_table(fact_relationship, "snomed_gold_fact_relationship")
            logger.info("Successfully completed Gold database load.")
        except Exception as e:
            logger.exception("Failed to load Gold database tables.")
            raise e
