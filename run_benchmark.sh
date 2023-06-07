#!/bin/zsh

module load open-ce/1.5.2-py38-0

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

#jsrun -n${NODES} -a6 -c42 -g6 -r1 --smpiargs "off" --bind=proportional-packed:7 python benchmark.py --tensor-model-parallel-size $1 --no-gradient-accumulation-fusion --seq_len 2048 --config configs/$2.yaml --recompute-activations --recompute-granularity selective --fp16
jsrun -n$3 -a6 -c42 -g6 -r1 --smpiargs "off" --bind=proportional-packed:7 python benchmark.py --tensor-model-parallel-size $1 --no-gradient-accumulation-fusion --seq_len 2048 --config configs/$2.yaml --recompute-activations --recompute-granularity selective --fp16 --batch-size-per-rank $4
#jsrun -n$3 -a6 -c42 -g6 -r1 --smpiargs "off" --bind=proportional-packed:7 python benchmark.py --tensor-model-parallel-size $1 --no-gradient-accumulation-fusion --seq_len 2048 --config configs/$2.yaml --fp16 --batch-size-per-rank $4
