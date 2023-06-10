import os
from functools import partial
print = partial(print, flush=True)

import math
from datetime import datetime
import subprocess
import yaml

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.initialize import set_jit_fusion_options
from megatron.arguments import parse_args
from megatron.global_vars import set_global_variables, set_args
from megatron.core.enums import ModelType
from megatron.model import GPTModel, Float16Module
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func


def estimate_params():
    l = args.num_layers
    h = args.hidden_size
    V = 2304
    s = args.seq_len
    return 12 * l * h ** 2 * (1 + 13 / (12 * h) + (V + s) / (12 * l * h))

def estimate_flops():
    l = args.num_layers
    h = args.hidden_size
    V = 2304
    s = args.seq_len
    B = args.batch_size_per_rank
    return 96 * B * s * l * h ** 2 * (1 + s / (6 * h) + V / (16 * l * h)) * mpu.get_data_parallel_world_size()


def main(args):
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', world_size=size, rank=rank)

    n_hier = args.n_hier
    mpu.set_n_hier(n_hier)
    largest_mp = -1
    for i in range(n_hier):
        mp = args.hier_tensor_model_parallel_size[i] * args.hier_pipeline_model_parallel_size[i]
        largest_mp = max(largest_mp, mp)
        mpu.initialize_model_parallel(args.hier_tensor_model_parallel_size[i], args.hier_pipeline_model_parallel_size[i], id=i)
    
    for i in range(n_hier):
        mp = args.hier_tensor_model_parallel_size[i] * args.hier_pipeline_model_parallel_size[i]
        assert largest_mp % mp == 0, (largest_mp, mp)


    mp0 = args.hier_tensor_model_parallel_size[0] * args.hier_pipeline_model_parallel_size[0]
    assert largest_mp % mp0 == 0
    n_chunks = largest_mp // mp0
        

    if rank == 0:
        for i in range(n_hier):
            print(f'HIER {i}')
            mpu.set_current_id(i)
            print(f'> initialized data model parallel with size '
                  f'{mpu.get_data_parallel_world_size()}')
            print(f'> initialized tensor model parallel with size '
                  f'{mpu.get_tensor_model_parallel_world_size()}')
            print(f'> initialized pipeline model parallel with size '
                  f'{mpu.get_pipeline_model_parallel_world_size()}')

    mpu.set_current_id(0)

#    set_jit_fusion_options()

    model = []
    for i in range(n_hier):
        pre_process = mpu.is_pipeline_first_stage() and i == 0
        post_process = mpu.is_pipeline_last_stage() and i == args.n_hiers - 1
        m = GPTModel(pre_process=pre_process, post_process=post_process)
        model.append(m)

    for m in model:
        for param in m.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    if rank == 0:
        params = sum([sum([p.nelement() for p in m.parameters()]) for m in model]) * mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
        params = params / 1e9
        print(f'> number of parameters: {params:.3f}B')

    for m in model:
        m.cuda(torch.cuda.current_device())

    if args.fp16:
        model = [Float16Module(m, args) for m in model]

    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = [torchDDP(m, device_ids=[i], output_device=i,
                        process_group=mpu.get_data_parallel_group()) for m in model]

    elif args.DDP_impl == 'local':
        model = [LocalDDP(m,
                            args.accumulate_allreduce_grads_in_fp32,
                            args.use_contiguous_buffers_in_local_ddp) for m in model]
        # broad cast params from data parallel src rank to other data parallel ranks
        # if args.data_parallel_random_init:
        #     for m in model:
        #         m.broadcast_params()
    else:
        raise NotImplementedError('Unknown DDP implementation specified: '
                                    '{}. Exiting.'.format(args.DDP_impl))

    optimizer = get_megatron_optimizer(model)

    for m in model:
        m.train()

    def random_batch():
        assert args.batch_size_per_rank % args.n_microbatches == 0
        b = args.batch_size_per_rank // args.n_microbatches
        batch = dict(
            tokens=torch.randint(0, 1024, (b, args.seq_len // n_chunks), dtype=torch.long),
            position_ids=torch.arange(args.seq_len // n_chunks, dtype=torch.long),
            attention_mask=None,
            labels=torch.randint(0, 1024, (b, args.seq_len // n_chunks), dtype=torch.long) 
        )
        batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
        return batch

    def loss_func(output_tensor):
        return output_tensor.mean(), {}

    def forward_step(data_iterator, model):
        batch = data_iterator
        return model(
            batch['tokens'], batch['position_ids'], batch['attention_mask'],
            labels=batch['labels']
        ), loss_func
    
    def train_step():
        batch = random_batch()

        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for m in model:
                m.zero_grad_buffer()
        optimizer.zero_grad()

        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch,
            model=model,
            num_microbatches=args.n_microbatches,
            dtype=args.params_dtype,
            tensor_shape=(args.seq_len // n_chunks, args.batch_size_per_rank // args.n_microbatches, args.hidden_size),
            grad_scaler=optimizer.scale_loss,
            sequence_parallel=args.sequence_parallel,
            overlap_p2p_comm=args.overlap_p2p_comm,
            batch_p2p_comm=not args.overlap_p2p_comm,
            forward_only=False,
            timers=None
        )
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        optimizer.reduce_model_grads(args, None)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, None)
        if update_successful:
            optimizer.gather_model_params(args, None)

        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

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
        print(f'Estimate params: {estimate_params()}')
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

    args = parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    ards_d = vars(args)
    ards_d.update(**config)

    args.padded_vocab_size = math.ceil(2304 / args.tensor_model_parallel_size) * args.tensor_model_parallel_size
    if args.fp16:
        args.params_dtype = torch.half
    else:
        args.params_dtype = torch.float
    args.model_type = ModelType.encoder_or_decoder
    #args.encoder_num_layers = args.num_layers
    #args.transformer_pipeline_model_parallel_size = args.pipeline_model_parallel_size
    args.virtual_pipeline_model_parallel_size = None
    set_args(args)

    #try:
    main(args)
    #except Exception:
    #    if rank == 0:
    #        print("Error")
