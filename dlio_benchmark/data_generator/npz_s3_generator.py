# dlio_benchmark/data_generator/npz_s3_generator.py

"""
NPZ S3 generator for DLIO Benchmark.
Builds each .npz in memory and uploads using the Rust binding.
"""

import time
import logging
import numpy as np
from io import BytesIO

from dlio_benchmark.data_generator.npz_generator import NPZGenerator
from dlio_benchmark.utils.config      import ConfigArguments
import dlio_s3_rust

# Use the same profiling decorator as base generator
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

class NPZS3Generator(NPZGenerator):
    """
    When generate_data=True and storage_type=='s3',
    marshal each NPZ in memory and push to S3.
    """

    def __init__(self):
        super().__init__()
        print("[DEBUG] NPZS3Generator initialized")

        cfg = ConfigArguments.get_instance()

        # pick up bucket and prefix from top-level config
        # bucket name and root prefix
        self.bucket = cfg.storage_root
        # base path inside bucket (data folder)
        self.base_folder = cfg.data_folder.rstrip('/')


    @dlp.log
    def save_file(self, out_path_spec: str, x, y):
        """
        out_path_spec: e.g. "train/img_000_of_168" (no extension)

        Builds an in-memory NPZ and uploads via Rust.
        """
        # 1) serialize to in-memory buffer and measure time
        buf = BytesIO()
        t0 = time.perf_counter()
        np.savez(buf, x=x, y=y)
        t1 = time.perf_counter()
        serialize_time = t1 - t0
        logging.info(f"[NPZS3] serialize npz in {serialize_time:.3f}s")
        data_bytes = buf.getvalue()

        # 2) form the final S3 URI
        print(f"[DEBUG] NPZS3Generator called with out_path_spec = {out_path_spec}")
        if out_path_spec.startswith("s3://"):
            # already a full URI
            uri = out_path_spec
        else:
            # make sure it has exactly one '.npz' at the end
            key = out_path_spec
            if not key.endswith(".npz"):
                key = key + ".npz"
            uri = f"s3://{self.bucket}/{key.lstrip('/')}"
        print(f"[DEBUG] NPZS3Generator.uploading to {uri!r} (bytes={len(data_bytes)})")


        # 3) upload bytes via Rust and measure time
        t2 = time.perf_counter()
        dlio_s3_rust.put_bytes(uri, data_bytes)
        t3 = time.perf_counter()
        upload_time = t3 - t2
        logging.info(f"[NPZS3] upload to {uri} in {upload_time:.3f}s")






