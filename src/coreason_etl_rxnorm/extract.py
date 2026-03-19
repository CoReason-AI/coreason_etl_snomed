# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import shutil
import zipfile
from pathlib import Path

from pydantic import BaseModel, Field

from coreason_etl_rxnorm.config import EpistemicRxNormPolicy
from coreason_etl_snomed.utils.logger import logger


class EpistemicBronzeExtractionIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for lossless extraction of the raw RRF
    files from the downloaded RxNorm ZIP archive to the local Bronze data layer,
    and subsequently cleaning up the temporary ZIP file.
    """

    policy: EpistemicRxNormPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the RxNorm pipeline.",
    )
    archive_path: Path = Field(
        ...,
        description="The path to the downloaded temporary ZIP archive containing RxNorm files.",
    )

    def execute(self) -> None:
        """
        Extracts specific RRF terminology files to the bronze data layer and
        removes the temporary archive.
        """
        bronze_dir = Path(self.policy.bronze_data_path)

        logger.info(f"Ensuring bronze directory exists at {bronze_dir}")
        bronze_dir.mkdir(parents=True, exist_ok=True)

        target_file_patterns = [
            "RXNCONSO.RRF",
            "RXNREL.RRF",
            "RXNSAT.RRF",
        ]

        logger.info(f"Extracting specific files from archive {self.archive_path}")
        extracted_files = []
        try:
            with zipfile.ZipFile(self.archive_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue

                    filename = Path(file_info.filename).name
                    normalized_path = "/" + file_info.filename.replace("\\", "/")
                    if "/rrf/" in normalized_path and filename in target_file_patterns:
                        target_path = bronze_dir / filename
                        with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        extracted_files.append(target_path)
                        logger.info(f"Extracted {filename} to {target_path}")
        except Exception as e:
            logger.exception("Failed to extract files from archive.")
            raise e
        finally:
            logger.info(f"Cleaning up temporary archive at {self.archive_path}")
            self.archive_path.unlink(missing_ok=True)

        if not extracted_files:
            logger.warning("No target RRF files were found in the archive.")
