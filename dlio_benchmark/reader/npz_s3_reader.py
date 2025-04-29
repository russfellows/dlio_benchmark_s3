"""
   NPZS3Reader: S3-backed reader for NPZ files using Rust storage.
   Overrides only open/close; indexing and sampling inherited from NPZReader.
"""
import numpy as np
import os
from io import BytesIO
from dlio_benchmark.reader.npz_reader import NPZReader
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import Profile
from abc import abstractmethod

# profiling decorator
dlp = Profile(MODULE_DATA_READER)

class NPZS3Reader(NPZReader):
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        # initialize base NPZReader (handles maps, args, logging)
        super().__init__(dataset_type, thread_index, epoch)

        print(f"[DEBUG] NPZS3Reader called with dataset_type = {dataset_type}")

        # fetch configuration
        cfg = ConfigArguments.get_instance()

        # pick up bucket and prefix from top-level config
        # bucket name and root prefix
        self.bucket = cfg.storage_root
        # base path inside bucket (data folder)
        self.base_folder = cfg.data_folder.rstrip('/')
        # override storage: S3-backed
        self.store = StorageFactory.get_storage(
            cfg.storage_type,
            cfg.storage_root,
            cfg.framework
        )

    @dlp.log
    def open(self, filename):
        """
        Load the NPZ file from S3 and return the 'x' array.
        """
        #print(f"[DEBUG] NPZS3Reader.open with filename = {filename}", flush=True)

        # Our filename is likely a jumbled mess, lets fix it
        # by stripping everything up to the base_folder
        split_token = self.base_folder + '/'
        parts = filename.rsplit(split_token, 1)
        #print(f"[DEBUG] NPZS3Reader after split our parts= {parts}")
        if len(parts) == 2:
            rel = parts[1]
        else:
            # fallback: use filename as-is
            rel = filename

        # Construct the storage-relative path under the bucket
        # make sure it has exactly one '.npz' at the end
        if not rel.endswith(".npz"):
            rel = rel + ".npz"
        #uri = f"s3://{self.bucket}/{self.base_folder}/{rel}"
        key = f"{self.base_folder}/{rel}"

        # Read raw NPZ bytes from S3
        print(f"[DEBUG] NPZS3Reader calling self.store.read with key = {key}", flush=True)
        raw = self.store.read(key)
        s3_size = len(raw)
        print(f"[DEBUG_STATS][NPZS3Reader.open pid={os.getpid()}] Get returned: {s3_size} bytes", flush=True)

        # Update stats 
        dlp.update(image_size=s3_size)

        # Load and return the 'x' array
        return np.load(BytesIO(raw), allow_pickle=True)['x']


    @dlp.log
    def close(self, filename):
        # delegate any base cleanup
        super().close(filename)
        # no additional resources to free
        pass

    # reuse NPZReader.get_sample and NPZReader.read_index
    # NPZReader.is_index_based / is_iterator_based reflect both modes

    @dlp.log
    def get_sample(self, filename, sample_index):
        return super().get_sample(filename, sample_index)

    @dlp.log
    def read_index(self, idx, step):
        return super().read_index(idx, step)

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

