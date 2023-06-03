#!/bin/zsh

module load open-ce/1.5.2-py38-0

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

jsrun -n${NODES} -a6 -c42 -g6 -r1 --smpiargs "off" --bind=proportional-packed:7 python benchmark.py --tensor-model-parallel-size $1 --num-layers 8 --hidden-size 1024 --ffn-hidden-size 4096 --num-attention-heads 16  --kv-channels 64 --no-gradient-accumulation-fusion --seq_len 2048
#jsrun -n1 -a1 -c42 -g1 -r1 --smpiargs "off" --bind=proportional-packed:7 python benchmark.py --tensor-model-parallel-size $1 --num-layers 8 --hidden-size 1024 --ffn-hidden-size 4096 --num-attention-heads 16  --kv-channels 64 --no-gradient-accumulation-fusion --seq_len 2048
