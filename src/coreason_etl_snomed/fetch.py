# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_snomed

import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.utils.logger import logger


class EpistemicOntologyFetchIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is temporarily modified to fetch a local SNOMED CT 
    archive instead of downloading it from the NLM API. It copies the local file to a 
    temporary secure file to prevent downstream deletion of the source asset.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self) -> Path:
        """
        Executes the fetch intent by sourcing a local SNOMED zip file.

        Returns:
            Path: The path to the copied temporary ZIP file.
        """
        logger.info("BYPASSING API FETCH: Using local snomed_data.zip for development.")
        
        # Point to your local zip file in the root directory
        local_zip_path = Path("snomed_data.zip")
        
        if not local_zip_path.exists():
            raise FileNotFoundError(f"Local archive not found at: {local_zip_path.absolute()}")

        try:
            # Create a temporary file to copy the zip into. 
            # This protects your original snomed_data.zip from being deleted by the extractor.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                temp_file_path = Path(temp_file.name)
                logger.info(f"Copying local {local_zip_path} to temporary workspace {temp_file_path}")
                
                shutil.copy(local_zip_path, temp_file_path)
                
            logger.info(f"Successfully prepared local archive at {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.exception("Failed to prepare local SNOMED CT archive.")
            if "temp_file_path" in locals():
                temp_file_path.unlink(missing_ok=True)
            raise e
