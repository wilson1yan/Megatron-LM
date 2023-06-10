import yaml
import argparse
import sys


def estimate_params():
    l = args.num_layers
    h = args.hidden_size
    V = 2048
    s = args.seq_len
    return 12 * l * h ** 2 * (1 + 13 / (12 * h) + (V + s) / (12 * l * h))

def estimate_flops():
    l = args.num_layers
    h = args.hidden_size
    V = 2048
    s = args.seq_len
    B = args.batch_size_per_rank
    return 96 * B * s * l * h ** 2 * (1 + s / (6 * h) + V / (16 * l * h)) * mpu.get_data_parallel_world_size()


def main(args):
    print(f'Estimate params: {estimate_params()}') 

if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1], 'r'))
    args = argparse.Namespace(**cfg)
    args.seq_len = 2048
    print(estimate_params() / 1e9)
