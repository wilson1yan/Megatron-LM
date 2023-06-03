import os
from datetime import datetime
import subprocess

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.initialize import set_jit_fusion_options
from megatron.arguments import parse_args
from megatron.global_vars import set_global_variables, set_args
from megatron.core.enums import ModelType
from megatron.model import GPTModel
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func


def main(args):
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', world_size=size, rank=rank)

    mpu.initialize_model_parallel(args.tensor_model_parallel_size)
    if rank == 0:
        print(f'> initialized data model parallel with size '
              f'{mpu.get_data_parallel_world_size()}')
        print(f'> initialized tensor model parallel with size '
              f'{mpu.get_tensor_model_parallel_world_size()}')

#    set_jit_fusion_options()

    model = GPTModel()
    for param in model.parameters():
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()]), flush=True))

    model.cuda(torch.cuda.current_device())

    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = torchDDP(model, device_ids=[i], output_device=i,
                        process_group=mpu.get_data_parallel_group())

    elif args.DDP_impl == 'local':
        model = LocalDDP(model,
                            args.accumulate_allreduce_grads_in_fp32,
                            args.use_contiguous_buffers_in_local_ddp)
        # broad cast params from data parallel src rank to other data parallel ranks
        if args.data_parallel_random_init:
            model.broadcast_params()
    else:
        raise NotImplementedError('Unknown DDP implementation specified: '
                                    '{}. Exiting.'.format(args.DDP_impl))

    optimizer = get_megatron_optimizer([model])

    model.train()

    def random_batch():
        batch = dict(
            tokens=torch.randint(0, 1024, (args.batch_size_per_rank, args.seq_len), dtype=torch.long),
            position_ids=torch.arange(args.seq_len, dtype=torch.long),
            attention_mask=None,
            labels=torch.randint(0, 1024, (args.batch_size_per_rank, args.seq_len), dtype=torch.long) 
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
            model.zero_grad_buffer()
        optimizer.zero_grad()

        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch,
            model=model,
            num_microbatches=1,
            dtype=args.params_dtype,
            tensor_shape=(args.seq_length, args.micro_batch_size, args.hidden_size),
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
        print(f"Iters: {args.trial_iters}, Elapsed: {elapsed / args.trial_iters} s/itr")
            

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
    args.padded_vocab_size = 2048
    args.params_dtype = torch.float32
    args.model_type = ModelType.encoder_or_decoder
    args.virtual_pipeline_model_parallel_size = None
    args.encoder_num_layers = args.num_layers
    set_args(args)
    main(args)
