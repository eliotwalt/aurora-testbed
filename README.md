# Aurora testbed

Memory issues are solved by using an ngc container. 

### Build the container
Change the `#SBATCH` arguments to match your cluster. 
```bash
sbatch container/build_container.job
```

### Run toy training/inference
Change the `#SBATCH` arguments to match your cluster. 
```bash
sbatch {train|infer}.job $OPTION
```
This will automatically execute in the container. `OPTION` argument specifies the model (large or small), precision (autocast, 'bf16-mode', nothing) and the layers on which to apply gradient checkpointing (defaults to all). Check the scripts for more details.