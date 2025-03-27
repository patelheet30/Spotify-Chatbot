import logging
import os
import tempfile
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return False


def download_image(url: str) -> Optional[str]:
    try:
        fd, temp_file_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(temp_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Image downloaded to {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None


def get_image_path(image_src: str) -> Optional[str]:
    if not image_src:
        logger.error("Image source is empty.")
        return None

    if is_url(image_src):
        logger.info("Image source is a URL.")
        return download_image(image_src)

    if os.path.exists(image_src):
        logger.info("Image source is a local file.")
        return image_src

    logger.error("Image source is neither a URL nor a valid file path.")
    return None
