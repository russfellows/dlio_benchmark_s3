# dlio_benchmark/checkpointing/s3_checkpoint_writer.py

import json
from io import BytesIO
import numpy as np

from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.utils.config import ConfigArguments
import dlio_s3_rust


class S3CheckpointWriter(BaseCheckpointing):
    """
    Implements the full BaseCheckpointing interface,
    streaming tensors and state dicts to S3 via Rust.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls(ext="bin")
        return cls._instance

    def __init__(self, ext="bin"):
        super().__init__(ext)
        cfg = ConfigArguments.get_instance()
        self.bucket = cfg.storage_root
        self.prefix = cfg.checkpoint_folder.rstrip("/")

    def checkpoint(self, epoch, step_number):
        """
        Called by DLIO to flush all of your layer+optimizer tensors.
        The base class has already gathered them into dicts:
          self.layer_state, self.optimization_state, self.model_state
        We'll package them into raw .npy buffers and upload in one shot.
        """
        checkpoint_id = f"epoch{epoch}_step{step_number}"
        buffers = []
        names = []

        # so get_tensor can locate this checkpoint
        self._last_checkpoint = checkpoint_id

        # layer tensors
        if self.layer_state:
            for name, arr in self.layer_state.items():
                buf = BytesIO()
                np.save(buf, arr)
                buffers.append(buf.getvalue())
                names.append(f"layer_{name}")

        # optimizer tensors
        if self.optimization_state:
            for name, arr in self.optimization_state.items():
                buf = BytesIO()
                np.save(buf, arr)
                buffers.append(buf.getvalue())
                names.append(f"optim_{name}")

        # model tensors
        if self.model_state:
            for name, arr in self.model_state.items():
                buf = BytesIO()
                np.save(buf, arr)
                buffers.append(buf.getvalue())
                names.append(f"model_{name}")

        uri_base = f"s3://{self.bucket}/{self.prefix}/{checkpoint_id}"
        self._upload_buffers(uri_base, buffers, names)

    def save_state(self, suffix, state, fsync=False):
        """
        Snapshot Python state (dicts, scalars, etc.), JSON-dump and upload to S3.
        """
        uri = f"s3://{self.bucket}/{self.prefix}/{suffix}.{self.ext}"
        data = json.dumps(state).encode("utf-8")
        dlio_s3_rust.put_bytes(uri, data)

    def get_tensor(self, size, datatype="int8"):
        """
        Download back exactly one of the npy buffers from the last checkpoint.
        We look at each slice in the .bin until we find one whose np.load().size==size,
        then return that array.
        """
        if not hasattr(self, "_last_checkpoint"):
            raise RuntimeError("No checkpoint has been written yet")

        uri_base = f"s3://{self.bucket}/{self.prefix}/{self._last_checkpoint}"
        idx_bytes = dlio_s3_rust.get_bytes(uri_base + ".idx")
        bin_bytes = dlio_s3_rust.get_bytes(uri_base + ".bin")

        # parse 8-byte little-endian offsets
        n = len(idx_bytes) // 8
        offsets = [
            int.from_bytes(idx_bytes[i*8:(i+1)*8], "little")
            for i in range(n)
        ]

        start = 0
        for end in offsets:
            chunk = bin_bytes[start:end]
            arr = np.load(BytesIO(chunk))
            if arr.size == size:
                return arr
            start = end

        raise KeyError(f"No tensor of size {size} found in checkpoint '{self._last_checkpoint}'")

    def finalize(self):
        """No-op for S3."""
        pass

    def _upload_buffers(self, uri_base, buffers, names):
        """
        buffers: list[bytes]
        names:   list[str]  # matching order
        Produces:
          uri_base + ".bin"   (multipart)
          uri_base + ".idx"   (8-byte offsets)
          uri_base + "_names.json"  (list of names)
        """
        bin_uri = uri_base + ".bin"
        dlio_s3_rust.multipart_put(bin_uri, buffers)

        # build byte-offset index
        offsets = []
        cursor = 0
        for b in buffers:
            cursor += len(b)
            offsets.append(cursor)
        idx_bytes = b"".join(o.to_bytes(8, "little") for o in offsets)
        dlio_s3_rust.put_bytes(uri_base + ".idx", idx_bytes)

        # upload the names list too
        names_data = json.dumps(names).encode("utf-8")
        dlio_s3_rust.put_bytes(uri_base + "_names.json", names_data)

