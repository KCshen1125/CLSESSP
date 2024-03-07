#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python evaluation.py \
    --model_name_or_path /hy-tmp/promptCSE-main/promptCSE-main/ConPVP/result/roberta-large:CLSESSP:66 \
                      --pooler 'mask' \
                      --mode 'test'

