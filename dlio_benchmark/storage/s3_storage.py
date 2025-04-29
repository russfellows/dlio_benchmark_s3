# dlio_benchmark/storage/s3_storage.py

"""
   Copyright (c) 2024, UChicago Argonne, LLC
   All Rights Reserved
   Licensed under the Apache License, Version 2.0 (the "License");
   ...
"""

from time import time
from io import BytesIO

from dlio_benchmark.common.constants    import MODULE_STORAGE
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType, StorageType
from dlio_benchmark.common.error_code   import ErrorCodes
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.utils.utility       import Profile

import os
import dlio_s3_rust as s3

dlp = Profile(MODULE_STORAGE)


class S3Storage(DataStorage):
    """
    Back-end that proxies every op to the Rust bindings.
    """

    @dlp.log
    def __init__(self, root, framework=None):
        super().__init__(framework)
        # root == bucket or bucket/prefix from storage_root
        if not root.startswith("s3://"):
            root = "s3://" + root.rstrip("/") + "/"
        self.root = root

    @dlp.log
    def _uri(self, path: str) -> str:
        return self.root + path.lstrip("/")

    # ---- Namespace APIs ----------------------------------------------------------

    @dlp.log
    def get_uri(self, id):  # required by DataStorage
        return self._uri(id)

    @dlp.log
    def create_namespace(self, exist_ok=False):
        # buckets are created out of band; nothing to do
        return True

    @dlp.log
    def get_namespace(self):
        return Namespace(self.root, NamespaceType.BUCKET)

    @dlp.log
    def create_node(self, id, exist_ok=False):
        # S3 has no concept of empty directory objects
        return True

    @dlp.log
    def get_node(self, id):
        # not implemented
        return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        return self.listdir(id)

    @dlp.log
    def delete_node(self, id):
        return self.delete(id)

    # ---- CRUD -------------------------------------------------------------

    @dlp.log
    def write(self, path: str, data: bytes):
        s3.put_bytes(self._uri(path), data)

    @dlp.log
    def read(self, path: str, offset=0, length=None) -> bytes:
        if offset == 0 and length is None:
            return bytes(s3.get_bytes(self._uri(path)))
        else:
            return bytes(s3.read_tensor_py(self._uri(path), offset, length))

    @dlp.log
    def listdir(self, path: str):
        return s3.list(self._uri(path))

    @dlp.log
    def exists(self, path: str) -> bool:
        try:
            s3.list(self._uri(path))
            return True
        except Exception:
            return False

    @dlp.log
    def delete(self, path: str):
        s3.delete(self._uri(path))

