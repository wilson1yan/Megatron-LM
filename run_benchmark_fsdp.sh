#!/bin/zsh

module load open-ce/1.5.2-py38-0

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

jsrun -n${NODES} -a6 -c42 -g6 -r1 --smpiargs "off" --bind=proportional-packed:7 python benchmark_fsdb.py --config configs/$1.yaml --batch_size_per_rank $2
