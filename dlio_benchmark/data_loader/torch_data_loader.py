"""
   Copyright (c) 2024, UChicago Argonne, LLC
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
import logging
import math
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import Sampler
import numpy as np
import os

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import Profile

# Add to count stats correctly 
from torch.utils.data._utils.collate import default_collate
from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_LOADER)

#
# Probably need to delete this function
#
def collate_and_profile(batch):
    # batch is a list of numpy arrays or torch.Tensors
    total_bytes = 0
    for item in batch:
        # if numpy array, .nbytes; if Tensor, convert to numpy
        if hasattr(item, "nbytes"):
            total_bytes += item.nbytes
        else:
            total_bytes += item.detach().cpu().numpy().nbytes

    # record I/O bytes and sample count in main
    print(f"[DEBUG_STATS][collate_and_profile pid={os.getpid()}] batch_size={len(batch)} total_bytes={total_bytes}")
    #  The next two lines should probably stay 
    dlp.update(image_size=total_bytes)
    dlp.update(step=len(batch))   # 
    #  I think these are bogus, probably delete next two lintes 
    #stats.update(image_size=total_bytes)
    #stats.update(step=len(batch))

    return default_collate(batch)


class TorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.num_samples = num_samples
        self.reader = None
        self.num_images_read = 0
        self.batch_size = batch_size
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.dlp_logger = None
        if num_workers == 0:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.dlp_logger = _args.configure_dftracer(is_child=True, use_pid=True)
        logging.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=worker_id,
                                               epoch_number=self.epoch_number)

    def __del__(self):
        if self.dlp_logger:
            self.dlp_logger.finalize()

    @dlp.log
    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        logging.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step = step)
        return self.reader.read_index(image_idx, step)


class dlio_sampler(Sampler):
    def __init__(self, rank, size, num_samples, epochs):
        self.size = size
        self.rank = rank
        self.num_samples = num_samples
        self.epochs = epochs
        samples_per_proc = int(math.ceil(num_samples/size)) 
        start_sample = self.rank * samples_per_proc
        end_sample = (self.rank + 1) * samples_per_proc - 1
        if end_sample > num_samples - 1:
            end_sample = num_samples - 1
        self.indices = list(range(start_sample, end_sample + 1))


    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for sample in self.indices:
            yield sample


class TorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)
    @dlp.log
    def read(self):
        dataset = TorchDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples,
                               self._args.read_threads, self.batch_size)
        sampler = dlio_sampler(self._args.my_rank, self._args.comm_size, self.num_samples, self._args.epochs)
        if self._args.read_threads >= 1:
            prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
        else:
            prefetch_factor = self._args.prefetch_size
        if prefetch_factor > 0:
            if self._args.my_rank == 0:
                logging.debug(
                    f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            prefetch_factor = 2
            if self._args.my_rank == 0:
                logging.debug(
                    f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        logging.debug(f"{utcnow()} Setup dataloader with {self._args.read_threads} workers {torch.__version__}")

        #
        # Do we ?   
        # We need to modify the threading and allow spawning
        # This is the code to do so if desired 
        #
        #import multiprocessing as mp
        #
        #if self._args.read_threads == 0:
        #    kwargs = {}
        #else:
        #    # create a fresh spawn‐based context
        #    spawn_ctx = mp.get_context("spawn")
        #    kwargs = {
        #        "multiprocessing_context": spawn_ctx,
        #        "prefetch_factor": prefetch_factor,
        #        "persistent_workers": (torch.__version__ != "1.3.1"),
        #    }

        #
        # Orig code, with forking mp 
        #
        if self._args.read_threads==0:
            kwargs={}
        else:
            kwargs={'multiprocessing_context':self._args.multiprocessing_context,
                    'prefetch_factor': prefetch_factor}
            if torch.__version__ != '1.3.1':       
                kwargs['persistent_workers'] = True

        #
        # End MP mods
        #

        #
        # Did have a call to "collate_and_profile" but is this nececssary?
        # Commented out the call in both branches of if below 
        #
        if torch.__version__ == '1.3.1':
            if 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=self._args.pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init, 
                                       #collate_fn=collate_and_profile,   # Added for multi threaded stats
                                       **kwargs)
        else: 
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=self._args.pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       #collate_fn=collate_and_profile,   # Added for multi threaded stats
                                       **kwargs)  # 2 is the default value
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataset) * self.batch_size} files")

        # self._dataset.sampler.set_epoch(epoch_number)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        step = 1
        # TODO: @hariharan-devarajan: change below line when we bump the dftracer version to 
        #       `dlp.iter(self._dataset, name=self.next.__qualname__)`

        #
        # Orig code to count batches
        #
        #for batch in dlp.iter(self._dataset):
        #    dlp.update(step = step)
        #    step += 1
        #    yield batch

        #
        # New Code for batches
        #
        for batch in dlp.iter(self._dataset):
            # 1) compute total bytes in this batch
            try:
                import numpy as _np
                batch_bytes = 0
                # assume batch is a tensor or tuple/list of arrays
                for x in batch if isinstance(batch, (list,tuple)) else (batch,):
                    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else _np.asarray(x)
                    batch_bytes += arr.nbytes
                dlp.update(image_size=batch_bytes)         # count I/O bytes
            except Exception:
                pass

            # 2) count the samples
            print(f"[DEBUG_STATS][next pid={os.getpid()}] yielding batch of {len(batch)} samples, step={step}")
            # Next line stays for now, maybe delete
            dlp.update(step=step)
            # Next line seems to be a problem 
            #stats.update(step=step)

            # Now update counter
            step += 1
            yield batch
     
        #
        # End code mod for batch counting
        #

        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        pass



