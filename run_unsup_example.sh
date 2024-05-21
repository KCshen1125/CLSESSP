#!/bin/bash
# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path path_to_your_model \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/CLSESSP-bb:seed42 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
    --pooler_type 'mask' \
    --prompt_len 4 \
    --do_train \
    --seed 42 \
    --do_eval \
    --fp16 \
    "$@"
