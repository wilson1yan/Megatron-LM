import argparse 
import yaml
from tfm_flops import *


V100_A100_ratio = 8
DEVICE_FLOPS = 383 * 1e12

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
    return 96 * B * s * l * h ** 2 * (1 + s / (6 * h) + V / (16 * l * h)) #* mpu.get_data_parallel_world_size()

def estimate_flops2():
    l = args.num_layers
    h = args.hidden_size
    V = 2048
    s = args.seq_len
    B = args.batch_size_per_rank
    return 4 * flops_tfm(h, s, None, l) * B


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-o', '--override_mp_devices', type=int, default=None)
#parser.add_argument('-b', '--batch_size_per_rank', type=int, required=True)
parser.add_argument('-b', '--total_batch_size', type=int, default=2048)
parser.add_argument('-u', '--flops_utilization', type=float, default=40.)
parser.add_argument('-s', '--seq_len', type=int, default=2048)
parser.add_argument('-i', '--iterations', type=float, default=2e6)
args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r'))
args_d = vars(args)
args_d.update(config)

params  = estimate_params()
print(f'Params: {params / 1e9:.3f}B')

if args.override_mp_devices is not None:
    a100 = args.override_mp_devices
else:
    a100 = args.mp_devices / V100_A100_ratio
flops = estimate_flops() * args.iterations
train_time = flops / (args.flops_utilization / 100. * DEVICE_FLOPS * a100)
train_time = train_time / 3600 / 24
print(f'FLOPS/itr: {estimate_flops():.4e} FLOPs, Total FLOPs: {flops:.4e} FLOPs')
print(f"Other {estimate_flops2():.4e} FLOPs")
print(f'Estimated train time for {args.iterations} itrs: {train_time:.3f} days')

total_devices = a100 * args.total_batch_size // args.batch_size_per_rank
total_nodes = total_devices / 4
total_node_hours = total_nodes * train_time * 24
print(f'Total devices: {a100 * args.total_batch_size // args.batch_size_per_rank}')
print(f'Total nodes: {total_nodes}, node hours: {total_node_hours}')
