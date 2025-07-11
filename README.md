# Aurora test bed
Training and inference of Aurora model on NVIDIA H100 GPUs. 

## Usage
Current scripts are specific to the Snellius HPC but changing the SLURM flags in `train_*.sh` and `infer_*.sh`.

### Create environment on a H100 node
```bash
sbatch make_env.sh
```
### Train
Training scripts are `train_*.sh`
```bash
# General usage
sbatch train_*.sh $option
```
with `option` an integer controlling what mixed precision and gradient checkpointing to use.

### Inference 
Inference scripts are `infer_*.sh`
```bash
# General usage
sbatch infer_*.sh $option
```
with `option` an integer controlling what mixed precision and gradient checkpointing to use.

### Run all experiemnts
The `run_experiments_*.sh` scripts call the training and inference scripts with all the possible option values automatically for convenenience. 


## Summary
- We are able to run stable inference.
- We are unable to train the large model with more than 1 GPU. We either get illegal memory access or cuda OOM errors, regardless of the configuration of mixed precision and activation checkpointing. 
- Our main observations are:
    - Inference should be run on independent processes (i.e. no torch distributed)
    - The mixed precision options do not work.
    - The small model consumes more memory than the large model.
    - There seems to be some GPU memory leaks somewhere in the model implementation as the memory usage increases, inevitably resulting in an OOM.

## Training results (DDP 1 GPU)
Script `train_1gpu.sh`. Logs can be found in `logs/train/1gpu`.

Note: `N=5` in `train.py`
| `option` | `job_id` |`num_epochs` | `small` | `bf16` | `autocast` |  `checkpointing_module_names`  | Error | When |
|-|-|-|-|-|-|-|-|-|
|0|13098829|500|true|false|false|all|None|NA|
|1|13098831|500|true|true|false|all|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|
|2|13098833|500|true|false|true|all|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|
|3|13098835|500|true|true|true|all|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|
|4|13098837|500|true|false|true|Basic3DEncoderLayer Basic3DDecoderLayer|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|
|5|13098840|500|false|false|false|all|Fail without error message. Loss is NaN though. Likely OOM.|During 33rd steps (epoch 7, batch 3)|
|6|13098842|500|false|true|false|all|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|
|7|13098860|500|false|false|true|all|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|
|8|13098862|500|false|true|true|all|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|
|9|13098866|500|false|false|true|Basic3DEncoderLayer Basic3DDecoderLayer|[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.|`loss.backward()` at first batch of first epoch|

### Observations
- Regardless of the number of parameters (small or large model), **the mixed precision options simply do not work**. 
- Option 5 is strange. The fact that we can perform 32 training steps with the large model without any mixed precision and suddenly fail indicates that there might be some **GPU memory accumulation somewhere in the model**.

### What can we do?
- Move to FSDP, PP, TP? 
- How did Microsoft manage?

## Training results (1 GPU, no DDP)
Script `train_1gpu_no_ddp.py`. Logs can be found in `logs/train/1gpu`

- `bf16-mode` triggers the illegal memory access error. (13102934.log)
- `autocast` triggers the illegal memory access error. (13102936.log)
- FP32 runs but will probably fail at some point (13102938.log)

## Training results (2 GPUs)

- `bf16-mode` triggers (13102221.log)
    - rank 0: Cuda OOM in loss.backward of batch 1 
    - rank 1: `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling 'cublasCreate(handle)'` (in forward of batch 1) and Cuda OOM (in forward of batch 1)
- `autocast` triggers (13102223.log)
    - rank 0: Cuda OOM in forward of batch 1 and illegal memory access error in loss.backward of batch 1
    - rank 1: Cuda OOM in forward of batch 1 and illegal memory access error in loss.backward of batch 1
- FP32
    - rank 0: Cuda OOM twice in forward
    - rank 1: Cuda OOM once in forward

## Observations
- Could the multiple errors in a snigle rank be due to gradient checkpointing? Because we recompute lots of things?



## Inference results (1 GPU)
Script `infer_1gpu.sh`. Logs can be found in `logs/infer/1gpu`.

Note: `N=100` in `infer.py`.

### Observations
- Overall, inference on 1 GPU seems to work.
- `bf16` + `autocast` + `small` does not work.
- GPU resources utilisation is strange. The small model uses more memory?
    - **Small model snapshot**
    ![Small model GPU snapshot](nsmi_inference_small.png)
    - **Large model snapshot**
    ![Large model GPU snapshot](nsmi_inference_large.png)

## Inference results (2GPUs)
Script `infer_2gpus.sh`. Logs can be found in `logs/infer/2gpus`.

- `bf16` fails. It raises `ChildFailedError` which does not provide any context. (13102222.log)
- `autocast` is slow: ~2it/s angainst ~1it/s in 1GPU mode (i.e. infer/1gpu/13098861.log) (13102224.log)
- FP32 fails. It raises `ChildFailedError` which does not provide any context. (13102222.log)

### Observations
- SLOWER than 1GPU and does not work!
- Again, it breaks after many steps, which could indicate some **memory accumulation!**
- We should just run multiple independent scripts. 

## Known issues
```
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &fbeta, c, CUDA_R_16BF, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
```
**Solution:** Add `export CUDA_LAUNCH_BLOCKING=1` to SLURM script.