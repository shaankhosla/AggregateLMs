#!/bin/bash

cd transformers/examples/pytorch/text-classification/

pip install datasets
export TASK_NAME=mnli

output_dir="ds_results"

batch_size=8

python3 run_glue.py \
  --model_name_or_path microsoft/deberta-v3-small \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 256 \
  --warmup_steps 1500 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 4.5e-5 \
  --num_train_epochs 3 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir $output_dir
