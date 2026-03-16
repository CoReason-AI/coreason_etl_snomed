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


class EpistemicSilverConceptIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for cleaning, typing, and filtering the raw Concept
    Snapshot terminology files from the local Bronze data layer into the Silver memory space,
    creating deterministic UUID5 identity resolutions.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self) -> pl.LazyFrame:
        """
        Executes the silver concept transformation. It scans the extracted Concept Snapshot files,
        applies filters, casts data types, and computes the deterministic `coreason_id`.

        Returns:
            pl.LazyFrame: A lazy evaluated polars DataFrame containing the transformed concepts.
        """
        bronze_dir = Path(self.policy.bronze_data_path)
        logger.info(f"Looking for Concept Snapshot files in {bronze_dir}")

        concept_files = list(bronze_dir.glob("sct2_Concept_Snapshot_*.txt"))
        if not concept_files:
            error_msg = f"No Concept Snapshot files found in {bronze_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # In case there are multiple, we'll just scan all matched files.
        # polars `scan_csv` supports glob patterns directly or a list of paths.
        concept_file_paths = [str(f) for f in concept_files]
        logger.info(f"Scanning files: {concept_file_paths}")

        # Ensure we treat 'id' as Utf8 upfront to avoid precision loss.
        lf = pl.scan_csv(
            concept_file_paths,
            separator="\t",
            schema_overrides={"id": pl.String, "moduleId": pl.String, "definitionStatusId": pl.String},
        )

        namespace_uuid = uuid.UUID(self.policy.snomed_namespace_uuid)

        def generate_uuid(snomed_id: str) -> str:
            return str(uuid.uuid5(namespace_uuid, snomed_id))

        transformed_lf = lf.filter(pl.col("active") == 1).with_columns(
            pl.col("effectiveTime").cast(pl.String).str.to_date("%Y%m%d").alias("effectiveTime"),
            pl.col("id").map_elements(generate_uuid, return_dtype=pl.String).alias("coreason_id"),
        )

        logger.info("Successfully constructed Silver Concept LazyFrame.")
        return transformed_lf


class EpistemicSilverDescriptionIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for cleaning, typing, and filtering the raw Description
    Snapshot terminology files from the local Bronze data layer into the Silver memory space,
    creating deterministic UUID5 identity resolutions.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self) -> pl.LazyFrame:
        """
        Executes the silver description transformation. It scans the extracted Description Snapshot files,
        applies filters, casts data types, and computes the deterministic `coreason_id` for the description
        and the associated concept.

        Returns:
            pl.LazyFrame: A lazy evaluated polars DataFrame containing the transformed descriptions.
        """
        bronze_dir = Path(self.policy.bronze_data_path)
        logger.info(f"Looking for Description Snapshot files in {bronze_dir}")

        description_files = list(bronze_dir.glob("sct2_Description_Snapshot_*.txt"))
        if not description_files:
            error_msg = f"No Description Snapshot files found in {bronze_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        description_file_paths = [str(f) for f in description_files]
        logger.info(f"Scanning files: {description_file_paths}")

        # Ensure we treat all IDs as Utf8 upfront to avoid precision loss.
        lf = pl.scan_csv(
            description_file_paths,
            separator="\t",
            schema_overrides={
                "id": pl.String,
                "moduleId": pl.String,
                "conceptId": pl.String,
                "languageCode": pl.String,
                "typeId": pl.String,
                "term": pl.String,
                "caseSignificanceId": pl.String,
            },
        )

        namespace_uuid = uuid.UUID(self.policy.snomed_namespace_uuid)

        def generate_uuid(snomed_id: str) -> str:
            return str(uuid.uuid5(namespace_uuid, snomed_id))

        transformed_lf = lf.filter(pl.col("active") == 1).with_columns(
            pl.col("effectiveTime").cast(pl.String).str.to_date("%Y%m%d").alias("effectiveTime"),
            pl.col("id").map_elements(generate_uuid, return_dtype=pl.String).alias("coreason_id"),
            pl.col("conceptId").map_elements(generate_uuid, return_dtype=pl.String).alias("concept_coreason_id"),
        )

        logger.info("Successfully constructed Silver Description LazyFrame.")
        return transformed_lf


class EpistemicSilverRelationshipIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for cleaning, typing, and filtering the raw Relationship
    Snapshot terminology files from the local Bronze data layer into the Silver memory space,
    creating deterministic UUID5 identity resolutions.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self) -> pl.LazyFrame:
        """
        Executes the silver relationship transformation. It scans the extracted Relationship Snapshot files,
        applies filters, casts data types, and computes the deterministic `coreason_id` for the relationship,
        source, destination, and type.

        Returns:
            pl.LazyFrame: A lazy evaluated polars DataFrame containing the transformed relationships.
        """
        bronze_dir = Path(self.policy.bronze_data_path)
        logger.info(f"Looking for Relationship Snapshot files in {bronze_dir}")

        relationship_files = list(bronze_dir.glob("sct2_Relationship_Snapshot_*.txt"))
        if not relationship_files:
            error_msg = f"No Relationship Snapshot files found in {bronze_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        relationship_file_paths = [str(f) for f in relationship_files]
        logger.info(f"Scanning files: {relationship_file_paths}")

        # Ensure we treat all IDs as Utf8 upfront to avoid precision loss.
        lf = pl.scan_csv(
            relationship_file_paths,
            separator="\t",
            schema_overrides={
                "id": pl.String,
                "moduleId": pl.String,
                "sourceId": pl.String,
                "destinationId": pl.String,
                "relationshipGroup": pl.String,
                "typeId": pl.String,
                "characteristicTypeId": pl.String,
                "modifierId": pl.String,
            },
        )

        namespace_uuid = uuid.UUID(self.policy.snomed_namespace_uuid)

        def generate_uuid(snomed_id: str) -> str:
            return str(uuid.uuid5(namespace_uuid, snomed_id))

        transformed_lf = lf.filter(pl.col("active") == 1).with_columns(
            pl.col("effectiveTime").cast(pl.String).str.to_date("%Y%m%d").alias("effectiveTime"),
            pl.col("id").map_elements(generate_uuid, return_dtype=pl.String).alias("coreason_id"),
            pl.col("sourceId").map_elements(generate_uuid, return_dtype=pl.String).alias("source_coreason_id"),
            pl.col("destinationId")
            .map_elements(generate_uuid, return_dtype=pl.String)
            .alias("destination_coreason_id"),
            pl.col("typeId").map_elements(generate_uuid, return_dtype=pl.String).alias("type_coreason_id"),
        )

        logger.info("Successfully constructed Silver Relationship LazyFrame.")
        return transformed_lf


class EpistemicGoldConceptIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for joining the Silver Concept and Silver Description
    data layers to create a denormalized, graph-ready representation of the concept.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self, silver_concept: pl.LazyFrame, silver_description: pl.LazyFrame) -> pl.LazyFrame:
        """
        Executes the gold concept transformation. It joins the Silver Concept with the Silver Description,
        filters for the Fully Specified Name, and assigns the primary string name.

        Args:
            silver_concept (pl.LazyFrame): The transformed Silver Concept data.
            silver_description (pl.LazyFrame): The transformed Silver Description data.

        Returns:
            pl.LazyFrame: A lazy evaluated polars DataFrame containing the transformed gold concepts.
        """
        logger.info("Constructing Gold Concept LazyFrame.")

        # Filter descriptions for Fully Specified Name (typeId == 900000000000003001) and active
        # active == 1 is already handled by Silver, but doing it again as defensive filtering
        # The primary string name should be just 'term'
        filtered_desc = silver_description.filter(
            (pl.col("typeId") == "900000000000003001") & (pl.col("active") == 1)
        ).select(["concept_coreason_id", "term"])

        # Join concept with filtered descriptions
        # Using left join to retain concepts even if they lack a Fully Specified Name for some reason
        transformed_lf = silver_concept.join(
            filtered_desc, left_on="coreason_id", right_on="concept_coreason_id", how="left"
        ).rename({"term": "name"})

        logger.info("Successfully constructed Gold Concept LazyFrame.")
        return transformed_lf


class EpistemicGoldSynonymIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for filtering the Silver Description data for
    acceptable synonyms and creating a bridge mapping between a concept's `coreason_id`
    and an array of its synonyms.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self, silver_description: pl.LazyFrame) -> pl.LazyFrame:
        """
        Executes the gold synonym transformation. It filters the Silver Description for acceptable synonyms
        and groups them into an array per concept.

        Args:
            silver_description (pl.LazyFrame): The transformed Silver Description data.

        Returns:
            pl.LazyFrame: A lazy evaluated polars DataFrame containing the transformed gold synonyms.
        """
        logger.info("Constructing Gold Synonym LazyFrame.")

        # Filter for acceptable synonyms (typeId == 900000000000013009) and active == 1
        filtered_desc = silver_description.filter((pl.col("typeId") == "900000000000013009") & (pl.col("active") == 1))

        # Group by concept_coreason_id and aggregate terms into a list, sorting to ensure determinism
        transformed_lf = (
            filtered_desc.group_by("concept_coreason_id")
            .agg(pl.col("term").sort().alias("synonyms"))
            .rename({"concept_coreason_id": "coreason_id"})
        )

        logger.info("Successfully constructed Gold Synonym LazyFrame.")
        return transformed_lf
