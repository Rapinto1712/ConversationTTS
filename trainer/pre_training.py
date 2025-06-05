"""
Author: Dongchao. 2025
The pre-training function for audio-text LLM
"""
import os
import time
import math
import pickle
import numpy as np
import inspect
import torch
import argparse
import logging
import json
import functools
from pathlib import Path
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn as nn 
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from utils.dataloader import get_data_iterator_tokenizer_vocabulary
import torch._dynamo
from models.model_new import ModelArgs, Model, CrossEntropyAndAccuracy_zero, CrossEntropyAndAccuracy_residual
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
from utils.train_utils import to_device
import contextlib
# Local dependency
from utils.train_utils import (
    seed_everything, 
    setup_logging,
    yaml_no_alias_safe_dump,
    save_checkpoint,
    maybe_resume_checkpoint,
    WarmupLR,
    str2bool,
    find_data_jsons,
    save_model
)
from utils.reporter import Reporter
from utils.arguments import get_args
import json
from safetensors.torch import load_model

def creat_optimizer_by_name(model, weight_decay, learning_rate, betas, global_rank, logger):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}

    # create optim groups.
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    # model params are flatten by fsdp, we need to set the params by its name
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(
        f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(
        f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(
        f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def get_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        base_lr: float = 1e-4,
        end_lr: float = 0.0,
):
    num_warmup_steps = int(num_training_steps * 0.03)
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model

def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")

def cuda_prefix_print(msg: str):
    prefix = f"[CUDA_{torch.cuda.current_device()}]"
    print(f"{prefix}\t{msg}")

def print_cuda_mem_info(msg: str):
    cuda_prefix_print(msg)
    cuda_prefix_print(f"[Allocated]\t{torch.cuda.max_memory_allocated()/1024:.1f}\tMiB")
    cuda_prefix_print(f"[Cached]\t{torch.cuda.max_memory_cached()/1024:.1f}\tMiB")
    cuda_prefix_print(f"[Reserved]\t{torch.cuda.max_memory_reserved()/1024:.1f}\tMiB")

def main():
    # (1) use DDP anyway (even for 1 GPU)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, local_rank, world_size = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    training_dtype = torch.bfloat16 # fix using torch.bfloat16
    # (2) arg parsing and logging
    args = get_args()
    args.local_rank = local_rank
    args.rank = rank
    if rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.exp_dir + '/logs', exist_ok=True)
    else:
        time.sleep(3)
    # =======================================
    #    Initialize wandb
    # =======================================
    timestamp = None
    if rank == 0:
        timestamp = time.localtime()
        timestamp = int(time.strftime("%Y%m%d%H%M%S", timestamp))
    # Convert timestamp to a tensor for broadcasting
    timestamp_tensor = torch.tensor([timestamp] if timestamp is not None else [0.0], dtype=torch.double).to(device)
    # Broadcast the timestamp to all processes
    dist.broadcast(timestamp_tensor, src=0)
    # All processes receive the timestamp
    timestamp = int(timestamp_tensor.item())
    
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128_256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )

    log_file = args.exp_dir + '/logs/RANK.log'
    setup_logging(rank, world_size, log_file)
    reporter = Reporter()
    # wandb
    if not args.no_wandb and rank == 0:
        os.environ["WANDB_DIR"] = experiment_dir
        wandb.init(
            project=args.wandb_project,
            name=f"{timestamp}-{model_string_name}",
            config=vars(args)
        )
    
    # (3) randomness & cudnn settings 
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    torch.manual_seed(1337 + args.seed)
    # build LLM and init the global transformer
    model = Model(config).to(device=device, dtype=training_dtype) # 
    
    # (4) data related objects: data iterator, tokenizers, vocabulary
    train_iter = get_data_iterator_tokenizer_vocabulary(
                    args=args,
                    train_jsons=find_data_jsons(args.train_data_jsons),
                    batch_scale=args.batch_scale,
                    minibatch_debug=args.minibatch_debug,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    n_worker=args.n_worker,
                    seed=args.seed,
                    parallel_number= args.parallel_number,
                    text_empty_token = 0,
                    semantic_empty_token = 0,
                    semantic_pad_token = args.pad_token,
                    semantic_eos = args.semantic_eos, 
                    text_pad_token= args.text_pad_token,
    ) 
    # (5) save config
    if rank == 0:
        with open(args.exp_dir + "/config.yaml", "w", encoding="utf-8") as f:
            logging.warning(f'Saving the configuration in {args.exp_dir}/config.yaml')
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)
    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )
    args.training_dtype = training_dtype # add training_dtype into args
    # (6) model, wrapped in FSDP
    model = setup_fsdp_sync(model, args, device)
    # (7) objects related to optimization: optimizer and scheduler model.out_norm.parameters()
    optimizer = creat_optimizer_by_name(model, args.weight_decay, args.learning_rate, (args.beta1, args.beta2), rank, logging)
    if args.schedule == 'cosine':
        total_train_steps = math.ceil(args.n_epoch * len(train_iter) / args.grad_accum)
        print('total_train_steps ', total_train_steps)
        scheduler = get_cosine_scheduler(optimizer, total_train_steps, base_lr=args.learning_rate)
    else:
        scheduler = None
        print('No scheduler?')
        assert args.schedule == None
    #scheduler = WarmupLR(optimizer, args.warmup_steps)
    # (8) Resume model, optimizer, scaler, etc, if needed.
    maybe_resume_checkpoint(args, model, optimizer, scheduler, reporter, train_iter)
    print(f'model arch: {model}')
    # statistics
    logging.info(f'model arch: {model}')
    # (9) training and evaluation
    start_epoch = reporter.get_epoch() + 1
    if start_epoch > args.n_epoch:
        logging.error(f'already reach the maximum training epochs. Done!')
    logging.info("training start ... ")
    print("training start ... ")
    for ep in range(start_epoch, args.n_epoch + 1):
        reporter.set_epoch(ep)
        # (10.1) train
        with reporter.observe("train") as sub_reporter:
            train_one_epoch(
              args=args,
              model=model,
              train_dl=train_iter,
              optimizer=optimizer,
              scheduler=scheduler,
              reporter=sub_reporter,
              parent_reporter=reporter,
            )
        train_iter.sampler.refresh() # refresh the dataloader
        # (10.2) evaluation
        # with torch.no_grad():
        #     with reporter.observe("valid") as sub_reporter:
        #         validate_model(
        #           args=args,
        #           model=model,
        #           valid_dl=train_iter,
        #           reporter=sub_reporter)
        # (10.3) epoch logging. 
        logging.info(reporter.log_message())
        # (10.4) save checkpoint
        checkpoint_path = args.exp_dir + f"/ep{ep}.checkpoint"
        logging.info(f"Saving checkpoint file {checkpoint_path}")
        #if ep % 5 ==0: # tmp code
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, reporter)

def train_one_epoch(args, model, train_dl, optimizer, scheduler, reporter, parent_reporter):
    model = model.train()
    optimizer.zero_grad()
    global_step = 0
    for b_idx, batch in enumerate(reporter.measure_iter_time(train_dl, "iter_time"), 1):
        global_step += 1
        seqs, masks, lengths, example_ids = batch
        data_stats = {"batch_size": len(seqs), "seq_len": seqs.size(1)}
        reporter.register(data_stats)
        with reporter.measure_time("forward_time"):
            with {
                "bf16": torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16),
                "fp16": torch.amp.autocast(device_type='cuda', dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext()}[args.mixed_precision]:
                c0_logits, ci_logits, ci_labels = model(tokens=seqs[:,:-1,:], labels=seqs[:,1:,:-1], tokens_mask=masks) # text_logits: B, T, D
                # print('c0_logits ', c0_logits.shape)
                # print('ci_logits ', ci_logits.shape)
                # print('reserved_mask ', reserved_mask.shape)
                # assert 1==2
                loss_0, metrics_0 = CrossEntropyAndAccuracy_zero(c0_logits, seqs[:,1:,0], masks[:,1:,0], ignore_id=args.pad_token)
                loss_residual, metrics_residual = CrossEntropyAndAccuracy_residual(ci_logits, ci_labels, loss_weights=[1.0]*(args.parallel_number-2), ignore_id=args.pad_token)
                loss = loss_0 + loss_residual # we set greater scale for the layer_0
                metrics = {}
                metrics['loss_tot'] = loss.clone().detach()
                metrics.update(metrics_0)
                metrics.update(metrics_residual)
                for v in metrics.values(): # Cross-GPU statistics
                    dist.all_reduce(v, dist.ReduceOp.AVG)
                reporter.register(metrics)
                #assert 1==2
            
        with reporter.measure_time("backward_time"):
            loss.backward()
        
        with reporter.measure_time("optim_time"):
            if b_idx % args.grad_accum == 0:
                grad_norm = model.clip_grad_norm_(args.grad_clip)
                if math.isnan(grad_norm):
                    logging.warning(f"grad norm is NaN. Discard this gradient")
                    optimizer.zero_grad()
                optimizer.step() # update the model even with ill gradient - sync the training
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                reporter.register({f'lr_param_{i}': pg['lr'] for i, pg in enumerate(optimizer.param_groups)})


        # must call this here so that the saved checkpoint is valid for reporter
        reporter.next()

        if b_idx % args.print_freq == 0:
            logging.info(reporter.log_message(-args.print_freq))
            if not args.no_wandb and args.rank == 0:
                wandb.log({"train_loss": metrics['loss_tot']}, step=global_step)

        if args.save_interval > 0 and global_step % args.save_interval == 0:
            checkpoint_path = args.exp_dir + f"/ep{reporter.get_epoch()}-iter{b_idx}.checkpoint"
            logging.info(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            #save_model(checkpoint_path, model)
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, parent_reporter)


def validate_model(args, model, valid_dl, reporter):
    model = model.eval()
    for b_idx, batch in enumerate(reporter.measure_iter_time(valid_dl, "iter_time"), 1):
        # batch = to_device(batch, "cuda")
        seqs, masks, lengths, example_ids = batch
        data_stats = {
            "batch_size": len(seqs),
            "seq_len": seqs.size(2),
        }
        reporter.register(data_stats)
        with reporter.measure_time("forward_time"):
            with {
                "bf16": torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16),
                "fp16": torch.amp.autocast(device_type='cuda', dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext()}[args.mixed_precision]:
                c0_logits, ci_logits = model(tokens=seqs[:,:-1,:], labels=seqs[:,1:,:-1], tokens_mask=masks) # text_logits: B, T, D
                loss_0, metrics_0 = CrossEntropyAndAccuracy_zero(c0_logits, seqs[:,1:,0], masks[:,1:,0], ignore_id=args.pad_token)
                loss_residual, metrics_residual = CrossEntropyAndAccuracy_residual(ci_logits, seqs[:,1:,1:-1], masks[:,1:,1:-1], loss_weights=[1.0]*(args.parallel_number-2), ignore_id=args.pad_token)
                loss = loss_0 + loss_residual # we set greater scale for the layer_0
                metrics = {}
                metrics['loss_tot'] = loss.clone().detach()
                metrics.update(metrics_0)
                metrics.update(metrics_residual)
                reporter.register(metrics)
        reporter.next()

if __name__ == '__main__':
    main()    

