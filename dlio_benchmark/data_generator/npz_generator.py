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
# File: dlio_benchmark/data_generator/npz_generator.py

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator

import logging
import numpy as np
import os

from dlio_benchmark.utils.utility import progress, utcnow
from dlio_benchmark.utils.utility import Profile
from shutil import copyfile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in NPZ format.
"""
class NPZGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    @dlp.log
    def save_file(self, out_path_spec: str, x, y):
        """
        Hook for writing one NPZ.
        By default: write to local disk at out_path_spec + '.npz'
        """
        path = out_path_spec + ".npz"
        dirname = os.path.dirname(path)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        if self.compression != Compression.ZIP:
            np.savez(path, x=x, y=y)
        else:
            np.savez_compressed(path, x=x, y=y)

    def generate(self):
        """
        Generator for creating data in NPZ format of 3d dataset.
        """
        super().generate()
        np.random.seed(10)
        record_labels = [0] * self.num_samples
        dim = self.get_dimension(self.total_files_to_generate)
        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim1 = dim[2*i]
            dim2 = dim[2*i+1]
            records = np.random.randint(255, size=(dim1, dim2, self.num_samples), dtype=np.uint8)

            out_path_spec = self.storage.get_uri(self._file_list[i])
            progress(i+1, self.total_files_to_generate, "Generating NPZ Data")

            # delegate the actual write to our hook
            self.save_file(out_path_spec, records, record_labels)

        # restore randomness
        np.random.seed()

