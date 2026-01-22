# Copied from https://github.com/Alpha-VLLM/Lumina-mGPT/blob/main/xllmx/data/data_reader.py

from io import BytesIO
import logging
import time
from typing import Union
import importlib
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


def read_general(path) -> Union[str, BytesIO]:
    if "s3://" in path:
        init_ceph_client_if_needed()
        file_bytes = BytesIO(client.get(path))
        return file_bytes
    else:
        return path


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        try:
            # from petrel_client.client import Client
            mod = importlib.import_module("petrel_client")
            Client = getattr(mod, "Client")
        except:
            raise RuntimeError("Failed to initialize ceph client")
        ed = time.time()

        client = Client("/path/to/petreloss.conf")
        logger.info(f"initialize client cost {ed - st:.2f} s")


client = None