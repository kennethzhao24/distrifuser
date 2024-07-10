#!/bin/bash

# N_GPUS=$1
# p=$2

# torchrun --nproc_per_node=$N_GPUS scripts/run_sdxl.py \
#    --mode benchmark \
#    --output_type latent \
#    --num_inference_steps 50 \
#    --image_size 1024 \
#    --parallelism $p


torchrun --nproc_per_node=4 scripts/test_dit_tp.py \
   --mode benchmark \
   --output_type latent \
   --num_inference_steps 50 \
   --image_size 1024 \
   --parallelism tensor \
   --no_cuda_graph


