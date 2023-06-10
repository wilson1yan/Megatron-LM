import os
import argparse
from functools import partial
print = partial(print, flush=True)

from datetime import datetime
import subprocess
import yaml

import torch
import torch.distributed as dist

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.checkpoint import checkpoint_wrapper
from fairscale.nn.wrap import wrap, enable_wrap, auto_wrap, default_auto_wrap_policy


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
    B = args.batch_size_per_rank * dist.get_world_size()
    return 96 * B * s * l * h ** 2 * (1 + s / (6 * h) + V / (16 * l * h))

    
class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=0., batch_first=False)
        self.linear1 = torch.nn.Linear(d_model, d_model * 4)
        self.linear2 = torch.nn.Linear(d_model * 4, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        h = self.norm1(x)
        h = self.self_attn(h, h, h)[0]
        x = x + h

        h = self.norm2(x)
        h = self.linear2(self.act(self.linear1(h)))
        x = x + h
        return x

    
class Transformer(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.embed_in = torch.nn.Embedding(1024, d_model)
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransformerDecoderLayer(d_model, nhead))
        self.norm = torch.nn.LayerNorm(d_model)
        self.fc_out = torch.nn.Linear(d_model, 1024)
    
    def forward(self, x):
        x = x.t()
        x = self.embed_in(x)
        for block in self.blocks:
            x = checkpoint_wrapper(block)(x)
            x = block(x)
        x = self.fc_out(self.norm(x))
        x = x.permute(1, 2, 0) # LND -> NDL)
        return x


def main(args):
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', world_size=size, rank=rank)

    fsdp_params = dict(flatten_parameters=True, compute_dtype=torch.float16, reshard_after_forward=True) 
    model = Transformer(
        d_model=args.hidden_size, 
        nhead=args.num_attention_heads,
        num_layers=args.num_layers
    )
    model = model.half()
    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model = auto_wrap(
            model,
            auto_wrap_policy=partial(default_auto_wrap_policy, recurse=True, min_num_params=1e6)
        )
    model = model.to(device)
    if rank == 0:
        print(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    if rank == 0:
        params = sum([p.nelement() for p in model.parameters()]) * dist.get_world_size()
        params = params / 1e9
        print(f'> number of parameters: {params:.3f}B')

    model.train()
    def random_batch():
        b = args.batch_size_per_rank
        batch = torch.randint(0, 1024, (b, args.seq_len), dtype=torch.long)
        batch = batch.to(device)
        return batch
    
    def train_step():
        batch = random_batch()

        optim.zero_grad()
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch)
        loss.backward()
        optim.step()
        
    torch.cuda.reset_peak_memory_stats(device=device)
    for _ in range(args.warmup_iters):
        train_step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(args.trial_iters):
        train_step()
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 1000
    if rank == 0:
        time_per_itr = elapsed / args.trial_iters
        print(f"Iters: {args.trial_iters}, Elapsed: {time_per_itr} s/itr")
        b = torch.cuda.max_memory_allocated(device=device)
        print(f'Peak Memory Usage: {b / 2 ** 30:.2f}GB')
        tflops = estimate_flops() / 1e12 / time_per_itr
        tflops_per_device = tflops / dist.get_world_size()
        print(f'Estimate TFLOP/s: {tflops:.2f}, Per Device: {tflops_per_device:.2f}')
        print(f'Estimate MFU: {tflops_per_device / 60 * 100:.0f}%')
            

if __name__ == "__main__":
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if rank == 0:
        print('> Started at ', datetime.now())

    get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
    os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    os.environ['MASTER_PORT'] = "23456"
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size_per_rank', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--warmup-iters', type=int, default=10)
    parser.add_argument('--trial-iters', type=int, default=50)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    args_d = vars(args)
    args_d.update(config)
    
    # try:
    main(args)
    # except Exception:
    #     if rank == 0:
    #         print("Error")
