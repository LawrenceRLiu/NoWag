#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1 #uncomment to specify GPUs
enviroment="NoWAC-VQ"

models=("meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-70b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-70B"
)
patterns=(
    "unstructured"
    "4_8"
    "2_4"
)

for model in "${models[@]}"; do
    for pattern in "${patterns[@]}"; do
        echo "===========Pruning $model with $pattern pattern=========="
        cmd="python -u NoWAG.py compress=prune run_name=prune_${pattern} eval=ppl_only"
        #if our pattern is not unstructured, then get the N:M
        if [[ $pattern != "unstructured" ]]; then
            n=$(echo $pattern | cut -d'_' -f1)
            m=$(echo $pattern | cut -d'_' -f2)
            cmd="$cmd +compress.kwargs.pattern=[${n},${m}]"
        fi
        echo "running command: $cmd"
        conda run -n $enviroment --live-stream $cmd
    done
done