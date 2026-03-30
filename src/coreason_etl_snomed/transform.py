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
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.utils.logger import logger

NAMESPACE_OMOP = uuid.UUID("b3d2b3c0-4f5c-4f7f-8e41-0f4b3f1f3e0f")

def generate_uuid(omop_id: str) -> str:
    """Helper to deterministically hash OMOP IDs."""
    if omop_id is None:
        return None
    return str(uuid.uuid5(NAMESPACE_OMOP, str(omop_id)))


class EpistemicSilverConceptIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class maps OMOP CONCEPT.csv into the Silver memory space,
    filtering for active concepts and establishing deterministic identity.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the pipeline.",
    )

    def execute(self) -> pl.LazyFrame:
        bronze_dir = Path(self.policy.bronze_data_path)
        concept_file = bronze_dir / "CONCEPT.csv"
        
        if not concept_file.exists():
            error_msg = f"No CONCEPT.csv found in {bronze_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Athena OMOP exports are typically tab-separated
        lf = pl.scan_csv(
            concept_file,
            separator="\t",
            quote_char=None,
            schema_overrides={
                "concept_id": pl.String, 
                "valid_start_date": pl.String, 
                "invalid_reason": pl.String
            },
            ignore_errors=True
        )

        # In OMOP, active concepts have a null invalid_reason
        transformed_lf = (
            lf.filter(pl.col("invalid_reason").is_null())
            .with_columns(
                pl.col("valid_start_date").cast(pl.String).str.to_date("%Y%m%d", strict=False).alias("effectiveTime"),
                pl.col("concept_id").alias("source_id"),
                pl.col("concept_id").map_elements(generate_uuid, return_dtype=pl.String).alias("coreason_id"),
            )
        )

        logger.info("Successfully constructed Silver Concept LazyFrame from OMOP.")
        return transformed_lf


class EpistemicSilverDescriptionIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class maps OMOP CONCEPT_SYNONYM.csv into the Silver memory space.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the pipeline.",
    )

    def execute(self) -> pl.LazyFrame:
        bronze_dir = Path(self.policy.bronze_data_path)
        synonym_file = bronze_dir / "CONCEPT_SYNONYM.csv"
        
        if not synonym_file.exists():
            error_msg = f"No CONCEPT_SYNONYM.csv found in {bronze_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        lf = pl.scan_csv(
            synonym_file,
            separator="\t",
            quote_char=None,
            schema_overrides={"concept_id": pl.String, "concept_synonym_name": pl.String},
            ignore_errors=True
        )

        transformed_lf = (
            lf.with_columns(
                pl.col("concept_id").map_elements(generate_uuid, return_dtype=pl.String).alias("concept_coreason_id"),
                pl.col("concept_synonym_name").alias("term")
            )
        )

        logger.info("Successfully constructed Silver Description LazyFrame from OMOP.")
        return transformed_lf


class EpistemicSilverRelationshipIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class maps OMOP CONCEPT_RELATIONSHIP.csv into the Silver memory space.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the pipeline.",
    )

    def execute(self) -> pl.LazyFrame:
        bronze_dir = Path(self.policy.bronze_data_path)
        rel_file = bronze_dir / "CONCEPT_RELATIONSHIP.csv"
        
        if not rel_file.exists():
            error_msg = f"No CONCEPT_RELATIONSHIP.csv found in {bronze_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        lf = pl.scan_csv(
            rel_file,
            separator="\t",
            quote_char=None,
            schema_overrides={
                "concept_id_1": pl.String,
                "concept_id_2": pl.String,
                "relationship_id": pl.String,
                "invalid_reason": pl.String,
                "valid_start_date": pl.String
            },
            ignore_errors=True
        )

        def generate_rel_uuid(struct):
            c1 = str(struct["concept_id_1"])
            r = str(struct["relationship_id"])
            c2 = str(struct["concept_id_2"])
            return str(uuid.uuid5(NAMESPACE_OMOP, f"{c1}-{r}-{c2}"))

        # In OMOP, active relationships have a null invalid_reason
        transformed_lf = (
            lf.filter(pl.col("invalid_reason").is_null())
            .with_columns(
                pl.col("valid_start_date").cast(pl.String).str.to_date("%Y%m%d", strict=False).alias("effectiveTime"),
                pl.col("concept_id_1").map_elements(generate_uuid, return_dtype=pl.String).alias("source_coreason_id"),
                pl.col("concept_id_2").map_elements(generate_uuid, return_dtype=pl.String).alias("destination_coreason_id"),
                pl.col("relationship_id").map_elements(generate_uuid, return_dtype=pl.String).alias("type_coreason_id"),
                pl.struct(["concept_id_1", "relationship_id", "concept_id_2"]).map_elements(generate_rel_uuid, return_dtype=pl.String).alias("coreason_id"),
            )
        )

        logger.info("Successfully constructed Silver Relationship LazyFrame from OMOP.")
        return transformed_lf


class EpistemicGoldConceptIntent(BaseModel):
    """
    AGENT INSTRUCTION: Elevates OMOP Silver Concepts into the Gold Dimension model.
    """

    policy: EpistemicOntologyPolicy = Field(...)

    def execute(self, silver_concept: pl.LazyFrame, silver_description: pl.LazyFrame) -> pl.LazyFrame:
        logger.info("Constructing Gold Concept LazyFrame.")
        
        # 1. Aggregate synonyms from the description table (Addresses Suggestion 2)
        synonyms_lf = (
            silver_description.group_by("concept_coreason_id")
            .agg(pl.col("term").sort().alias("synonyms"))
        )

        # 2. Join the concept table with the aggregated synonyms
        transformed_lf = (
            silver_concept
            .join(
                synonyms_lf,
                left_on="coreason_id",
                right_on="concept_coreason_id",
                how="left"
            )
            .rename({"concept_name": "name"})
            # Fill concepts that have no synonyms with an empty list for consistency
            .with_columns(pl.col("synonyms").fill_null([]))
            # Hide redundant source_id (Addresses Suggestion 4) 
            # Retains all other native OMOP columns natively (Addresses Suggestion 1)
            .drop(["source_id"]) 
        )

        return transformed_lf


class EpistemicGoldRelationshipIntent(BaseModel):
    """
    AGENT INSTRUCTION: Shapes OMOP relationships into the Gold Fact mapping.
    """

    policy: EpistemicOntologyPolicy = Field(...)

    def execute(self, silver_relationship: pl.LazyFrame) -> pl.LazyFrame:
        logger.info("Constructing Gold Relationship LazyFrame.")

        transformed_lf = silver_relationship.select([
            pl.col("coreason_id"),
            pl.col("source_coreason_id"),
            pl.col("destination_coreason_id"),
            pl.col("type_coreason_id").alias("relationship_type_coreason_id"),
        ])

        return transformed_lf
