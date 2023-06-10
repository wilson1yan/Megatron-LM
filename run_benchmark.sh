#!/bin/zsh

module load open-ce/1.5.2-py38-0

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

jsrun -n${NODES} -a6 -c42 -g6 -r1 --smpiargs "off" --bind=proportional-packed:7 python benchmark.py --tensor-model-parallel-size $1 --no-gradient-accumulation-fusion --seq_len 2048 --config configs/$2.yaml --batch-size-per-rank $3 --n-microbatches $4 --pipeline-model-parallel-size $5  --fp16 --recompute-activations --recompute-granularity selective
