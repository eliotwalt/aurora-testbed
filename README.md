# Aurora testbed

Running Aurora on NVIDIA GPUs.

## Known issues
```
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &fbeta, c, CUDA_R_16BF, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
```
**Solution:** Add `export CUDA_LAUNCH_BLOCKING=1` to SLURM script.