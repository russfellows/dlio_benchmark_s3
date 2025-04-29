#!/bin/bash
mpirun  -hosts 127.0.0.1 -np 1 python ./dlio_benchmark/main.py  workload=unet3d_h100 ++workload.workflow.generate_data=False ++workload.workflow.train=True

