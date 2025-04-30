# Fork of DLIO Benchmark
Note: This is a fork of the DLIO benchmark, specifically for adding support of S3 storage.  

The main Readme for DLIO is available on that GitHub repo and in other on-line documentation.  This project will not repeat that information here.

## Deep Learning I/O (DLIO) Benchmark

The DLIO README in its github repository provides documentation of the DLIO code. Please refer to [DLIO Docs](https://dlio-benchmark.readthedocs.io) for full user documentation. 

## Overview

DLIO is an I/O benchmark for Deep Learning. DLIO is aimed at emulating the I/O behavior of various deep learning applications. The benchmark is delivered as an executable that can be configured for various I/O patterns. It uses a modular design to incorporate more data loaders, data formats, datasets, and configuration parameters. It emulates modern deep learning applications using Benchmark Runner, Data Generator, Format Handler, and I/O Profiler modules. 

# S3 Support
The intention of this fork is to test S3 access using DLIO.  This project [dlio_benchmark_s3] uses a custom S3 Python library, built using the AWS Rust S3 SDK, which is also available here: [dlio S3 Rust] (https://github.com/russfellows/dlio_s3_rust)

## Installation and running
You may of course run this directly as downloaded.  However, some executables are presumed to be in your path.  Due to the variety of paths, OS environments , etc. it is VASTLY more reliable and predictable to run this within a container. The container includes the dlio-s3-rust Python library and standalone cli for access to S3.

## Container
```bash
git clone https://github.com/russfellows/dlio_benchmark_s3
cd ./dlio_benchmark_s3
docker build -t dlio_benchmark_s3 .
docker run --rm --net=host -it dlio_benchmark_s3
``` 

### Starting a data generation the container
```bash
root@hostname:/workdir/dlio# mpirun  -hosts 127.0.0.1 -np 1 python ./dlio_benchmark/main.py  workload=unet3d_h100 ++workload.workflow.generate_data=True ++workload.workflow.train=False
```
### Starting a benchmark run in the container
```bash
root@hostname:/workdir/dlio# mpirun  -hosts 127.0.0.1 -np 1 python ./dlio_benchmark/main.py  workload=unet3d_h100 ++workload.workflow.generate_data=False ++workload.workflow.train=True
```

The configurations of a workload can be specified through a yaml file. Examples of yaml files can be found in [dlio_benchmark/configs/workload/](./dlio_benchmark/configs/workload). 

The full list of configurations can be found in: https://argonne-lcf.github.io/dlio_benchmark/config.html

The YAML file is loaded through hydra (https://hydra.cc/). The default setting are overridden by the configurations loaded from the YAML file. One can override the configuration through command line (https://hydra.cc/docs/advanced/override_grammar/basic/). 
