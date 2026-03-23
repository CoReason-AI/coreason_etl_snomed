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

import requests
from pydantic import BaseModel, Field

from coreason_etl_snomed.config import EpistemicOntologyPolicy
from coreason_etl_snomed.utils.logger import logger


class EpistemicOntologyFetchIntent(BaseModel):
    """
    AGENT INSTRUCTION: This class is responsible for fetching the latest SNOMED CT US Edition
    download URL from the NLM API and downloading the ZIP into a secure temporary file.
    """

    policy: EpistemicOntologyPolicy = Field(
        ...,
        description="The epistemological and configuration boundaries for the SNOMED pipeline.",
    )

    def execute(self) -> Path:
        """
        Executes the fetch intent, downloading the latest SNOMED release.

        Returns:
            Path: The path to the downloaded temporary ZIP file.
        """
        api_key = self.policy.umls_api_key.get_secret_value()

        # 1. Get Latest Release URL
        logger.info("Fetching latest SNOMED CT US Edition release URL.")
        release_url = "https://uts-ws.nlm.nih.gov/releases?releaseType=snomed-ct-us-edition&current=true"
        response = requests.get(release_url, timeout=30)
        response.raise_for_status()

        data = response.json()
        if not data or not isinstance(data, list) or "downloadUrl" not in data[0]:
            raise ValueError("Invalid response format from UTS Release API.")

        download_url = data[0]["downloadUrl"]
        logger.info(f"Obtained download URL: {download_url}")

        # 2. Download the Archive
        logger.info("Downloading SNOMED CT archive.")
        download_endpoint = f"https://uts-ws.nlm.nih.gov/download?url={download_url}&apiKey={api_key}"

        # We need to stream the download into a temp file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                temp_file_path = Path(temp_file.name)
                with requests.get(download_endpoint, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    # Ensure shutil.copyfileobj streams correctly by decoding raw response stream
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, temp_file)
            logger.info(f"Successfully downloaded archive to {temp_file_path}")
            return temp_file_path
        except Exception as e:
            logger.exception("Failed to download SNOMED CT archive.")
            if "temp_file_path" in locals():
                temp_file_path.unlink(missing_ok=True)
            raise e
