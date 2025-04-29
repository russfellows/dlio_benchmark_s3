# dlio_benchmark/data_generator/generator_factory.py

"""
   Copyright (c) 2024, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""

import logging

from dlio_benchmark.common.enumerations import FormatType, StorageType
from dlio_benchmark.common.error_code import ErrorCodes
from dlio_benchmark.utils.config import ConfigArguments

# stock generators
from dlio_benchmark.data_generator.tf_generator import TFRecordGenerator
from dlio_benchmark.data_generator.hdf5_generator import HDF5Generator
from dlio_benchmark.data_generator.csv_generator import CSVGenerator
from dlio_benchmark.data_generator.npz_generator import NPZGenerator
from dlio_benchmark.data_generator.npy_generator import NPYGenerator
from dlio_benchmark.data_generator.jpeg_generator import JPEGGenerator
from dlio_benchmark.data_generator.png_generator import PNGGenerator
from dlio_benchmark.data_generator.synthetic_generator import SyntheticGenerator
from dlio_benchmark.data_generator.indexed_binary_generator import IndexedBinaryGenerator

# S3-aware generators (assuming these exist and are named BaseClassNameS3Generator)
# These imports are conditional and might fail if the files don't exist,
# which is handled by checking in get_generator.
try:
    from dlio_benchmark.data_generator.tf_s3_generator import TFRecordS3Generator
    from dlio_benchmark.data_generator.hdf5_s3_generator import HDF5S3Generator
    # Assuming naming convention, adjust if your S3 classes have different names
    from dlio_benchmark.data_generator.csv_s3_generator import CSVs3Generator
    from dlio_benchmark.data_generator.npz_s3_generator import NPZS3Generator
    from dlio_benchmark.data_generator.npy_s3_generator import NPYs3Generator
    from dlio_benchmark.data_generator.jpeg_s3_generator import JPEGS3Generator
    from dlio_benchmark.data_generator.png_s3_generator import PNGS3Generator
    from dlio_benchmark.data_generator.synthetic_s3_generator import SyntheticS3Generator
    from dlio_benchmark.data_generator.indexed_binary_s3_generator import IndexedBinaryS3Generator
    _s3_generators_imported = True
except ImportError as e:
    # Log a warning if imports fail, but don't crash.
    # The factory will fall back to base generators.
    logging.getLogger(__name__).warning(f"Could not import all S3 generators. Falling back to base generators where S3 specific ones are missing. Error: {e}")
    _s3_generators_imported = False


# Map enum -> base generator class
_GEN_MAP = {
    FormatType.TFRECORD: TFRecordGenerator,
    FormatType.HDF5: HDF5Generator,
    FormatType.CSV: CSVGenerator,
    FormatType.NPZ: NPZGenerator,
    FormatType.NPY: NPYGenerator,
    FormatType.JPEG: JPEGGenerator,
    FormatType.PNG: PNGGenerator,
    FormatType.SYNTHETIC: SyntheticGenerator,
    FormatType.INDEXED_BINARY: IndexedBinaryGenerator,
    # MMAP_INDEXED_BINARY uses the same generator as INDEXED_BINARY for generation
    FormatType.MMAP_INDEXED_BINARY: IndexedBinaryGenerator,
}

# Map enum -> potential S3 generator class name string
# This allows looking up the class by name in globals()
_S3_GEN_NAME_MAP = {
    FormatType.TFRECORD: "TFRecordS3Generator",
    FormatType.HDF5: "HDF5S3Generator",
    FormatType.CSV: "CSVs3Generator", # Assuming this name
    FormatType.NPZ: "NPZS3Generator",
    FormatType.NPY: "NPYs3Generator", # Assuming this name
    FormatType.JPEG: "JPEGS3Generator", # Assuming this name
    FormatType.PNG: "PNGS3Generator", # Assuming this name
    FormatType.SYNTHETIC: "SyntheticS3Generator", # Assuming this name
    FormatType.INDEXED_BINARY: "IndexedBinaryS3Generator", # Assuming this name
    FormatType.MMAP_INDEXED_BINARY: "IndexedBinaryS3Generator", # Assuming this name
}


class GeneratorFactory:
    """
    Picks the right data generator based on format *and* storage_type.
    If storage is S3, it attempts to use an S3-aware generator subclass
    by appending '_s3' to the base class name, falling back if not available.
    """

    @staticmethod
    def get_generator(fmt):
        args = ConfigArguments.get_instance()

        logging.getLogger(__name__).debug(f"In get_generator: args.storage_type = {args.storage_type}, fmt = {fmt}")

        # 1) Normalize fmt to a FormatType enum
        if isinstance(fmt, FormatType):
            fmt_enum = fmt
        else:
            try:
                fmt_enum = FormatType[fmt.upper()]
            except KeyError:
                logging.getLogger(__name__).error(f"Unknown format '{fmt}'", exc_info=True)
                raise Exception(f"Unknown format '{fmt}'") from None

        selected_gen_cls = None

        # 2) Check if using S3 storage and if S3 generators were successfully imported
        if args.storage_type == StorageType.S3 and _s3_generators_imported:
            # Try to find the S3 generator class name for this format
            s3_gen_name = _S3_GEN_NAME_MAP.get(fmt_enum)

            if s3_gen_name:
                try:
                    # Attempt to get the S3 generator class from the current module's globals
                    selected_gen_cls = globals()[s3_gen_name]
                    logging.getLogger(__name__).info(f"Selected {s3_gen_name} for S3 + {fmt_enum.name}")
                except KeyError:
                    # S3 generator name was mapped, but the class wasn't found in globals()
                    # This happens if the specific S3 generator import failed, even if _s3_generators_imported is True
                     logging.getLogger(__name__).warning(
                         f"S3 generator class '{s3_gen_name}' not found or imported for format {fmt_enum.name}. "
                         f"Falling back to base generator."
                     )
                     selected_gen_cls = None # Ensure it's None so we fall back
            else:
                 # Format enum was not mapped to an S3 generator name
                 logging.getLogger(__name__).warning(
                     f"Format {fmt_enum.name} is not mapped to an S3 generator name in _S3_GEN_NAME_MAP. "
                     f"Falling back to base generator even though storage is S3."
                 )
                 # Fallback to base generator will happen below

        # 3) If an S3 generator wasn't selected, fallback to the base generator
        if selected_gen_cls is None:
            try:
                selected_gen_cls = _GEN_MAP[fmt_enum]
                # Log which base generator is being used and why (either not S3,
                # S3 generators not imported, or no specific S3 generator for this format)
                reason = "due to non-S3 storage"
                if args.storage_type == StorageType.S3:
                    reason = "as no specific S3 generator found/imported"
                logging.getLogger(__name__).info(f"Selected {selected_gen_cls.__name__} {reason} for format {fmt_enum.name}")

            except KeyError:
                # This should theoretically not happen if _GEN_MAP is complete,
                # but left for robustness from the original code.
                logging.getLogger(__name__).error(f"Base generator not found for format {fmt_enum.name}", exc_info=True)
                raise Exception(str(ErrorCodes.EC1001))

        # Instantiate and return the selected generator class
        # Assuming all generator constructors take no arguments based on the original code
        return selected_gen_cls()


