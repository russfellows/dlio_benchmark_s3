"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
"""
   Copyright (c) 2025, Signal65, Futurum LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from time import time

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os

from dlio_benchmark.utils.utility import Profile

######
# dlio_benchmark/storage/s3_storage.py`  was a stub: every method delegates to  `DataStorage`, which immediately returns  `None`
# This is why it doesn't work to perform operations like Get, Put and List. 
# For now, "List" operations are implemented as a minimal, boto3‑based implementation, since it is not performance critical
# 
# The new, backend Rust library will be used for the heavy data path I/O like Get and Put.
#
######
# New imports
import boto3, os, re
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType

dlp = Profile(MODULE_STORAGE)

# New definitions 
class S3RustStorage(DataStorage):
    def __init__(self, bucket, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(bucket, NamespaceType.FLAT)
        self._s3 = boto3.client("s3")

    # ---- metadata helpers used by DLIO ----
    def get_uri(self, key):                    # s3://bucket/obj
        return f"s3://{os.path.join(self.namespace.name, key)}"

    def walk_node(self, prefix, use_pattern=False):
        paginator = self._s3.get_paginator("list_objects_v2")
        prefix = prefix.lstrip("/")
        keys = []
        for page in paginator.paginate(Bucket=self.namespace.name,
                                       Prefix=prefix):
            keys += [c["Key"] for c in page.get("Contents", [])]
        if use_pattern:          # crude glob for “*/*.npz” etc.
            regex = re.compile(os.path.basename(prefix).replace("*", ".*"))
            keys = [k for k in keys if regex.match(os.path.basename(k))]
        return keys

    def get_node(self, key=""):
        return MetadataType.S3_OBJECT if key else MetadataType.DIRECTORY

    # create/delete/put/get can be stubs for read‑only benchmarking
    def create_namespace(self, exist_ok=False): return True
    def create_node(self, id, exist_ok=False):  return True
    def delete_node(self, id):                  return True


# Old Class definition, Not used?
class S3Storage(DataStorage):
    """
    Storage APIs for creating files.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

    @dlp.log
    def get_uri(self, id):
        return "s3://" + os.path.join(self.namespace.name, id)

    @dlp.log
    def create_namespace(self, exist_ok=False):
        return True

    @dlp.log
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @dlp.log
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @dlp.log
    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        return super().walk_node(self.get_uri(id), use_pattern)

    @dlp.log
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        return super().put_data(self.get_uri(id), data, offset, length)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        return super().get_data(self.get_uri(id), data, offset, length)

    def get_basename(self, id):
        return os.path.basename(id)



