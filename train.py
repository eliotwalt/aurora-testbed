import os
import gc
import datetime
import contextlib
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from aurora.model.aurora import Aurora, AuroraSmallPretrained
from aurora import Batch, Metadata

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# https://github.com/microsoft/aurora/issues/80#issuecomment-2849357112
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:32"

def make_batch(x, lat, lon):
    H, W = x.shape[-2:]
    return Batch(
        surf_vars={var: x[:,:,i] for i, var in enumerate(["10u", "10v", "2t", "msl"])},
        atmos_vars={var: x[:,:,i*13:(i+1)*13] for i, var in enumerate(["t", "u", "v", "q", "z"])},
        static_vars={var: torch.randn(H,W) for var in ["z", "lsm", "slt"]},
        metadata=Metadata(
            lat=lat, lon=lon, time=(datetime.datetime.now(),),
            atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        )
    )
    
def compute_loss(criterion, preds, target):
    loss = None
    for var_kind in ["surf_vars", "atmos_vars"]:
        for var_name in getattr(target, var_kind).keys():
            pred_var = getattr(preds, var_kind)[var_name]
            target_var = getattr(target, var_kind)[var_name]
            if loss is None: loss = criterion(pred_var, target_var)
            else: loss += criterion(pred_var, target_var)
    return loss

def custom_activation_checkpointing(model, module_names=("Perceiver3DEncoder","Swin3DTransformerBackbone","Basic3DEncoderLayer","Basic3DDecoderLayer","Perceiver3DDecoder","LinearPatchReconstruction")):
        def check(x: torch.nn.Module) -> bool:
            name = x.__class__.__name__
            return name in module_names
        apply_activation_checkpointing(model, check_fn=check)

def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='nccl')
    device = torch.device(f'cuda:{local_rank}')
    
    # Dummy dataset
    N = 5  # Number of samples
    M = 69 # 4 + 5 * 13
    H, W = 720, 1440
    lat=torch.linspace(90, -90, H)
    lon=torch.linspace(0, 360, W+1)[:-1]
    random_input = torch.randn(N, 2, M, H, W, dtype=torch.bfloat16)
    random_target = torch.randn(N, 1, M, H, W, dtype=torch.bfloat16)
    dataset = TensorDataset(random_input, random_target)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=1)
    print("Dummy dataset created with random data.")
    
    # model
    if args.small:
        model = AuroraSmallPretrained(bf16_mode=args.bf16, use_lora=False)
    else:
        model = Aurora(bf16_mode=args.bf16, use_lora=False)
    model.load_checkpoint(strict=False)
    custom_activation_checkpointing(model, args.checkpointing_module_names)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    print("Model loaded and wrapped in DDP.")
    
    # loss and optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, lr=5e-6)
    print("Loss and optimizer configured.")
    
    # autocast context
    if args.autocast: autocast_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else: autocast_context = contextlib.nullcontext()
    
    # training loop
    for epoch in range(args.num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(dataloader):
            gc.collect()
            torch.cuda.empty_cache()
            # move to device
            inputs, targets = map(lambda x: make_batch(x, lat, lon)._fmap(lambda x: x.to(device, non_blocking=True)), (inputs, targets))
            optimizer.zero_grad()
            # fwd
            with autocast_context:
                outputs = model(inputs)
                loss = compute_loss(criterion, outputs, targets)
            # bwd 
            loss.backward()
            optimizer.step()
            if local_rank == 0: print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
def main(args):
    try: train(args)
    except Exception as e: dist.destroy_process_group() ; raise e
    
if __name__ == "__main__":
    print("Starting Aurora BF16 training script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--small", action="store_true", help="Use the small version of the Aurora model")
    parser.add_argument("--bf16", action="store_true", help="Use `bf16_mode`")
    parser.add_argument("--autocast", action="store_true", help="Use autocast context with bf16")
    parser.add_argument("--checkpointing_module_names", type=str, nargs='+', 
                        default=["Perceiver3DEncoder","Swin3DTransformerBackbone","Basic3DEncoderLayer","Basic3DDecoderLayer","Perceiver3DDecoder","LinearPatchReconstruction"])
    args = parser.parse_args()
    main(args)
    
    