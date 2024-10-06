#!/bin/bash
# if [ $MASTER_ADDR ];then
# 	echo $MASTER_ADDR
#     echo $MASTER_PORT
#     echo $WORLD_SIZE
#     echo $RANK
# else
# 	MASTER_ADDR=127.0.0.1
#     MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
#     WORLD_SIZE=1
#     RANK=0
# fi
# # Change for multinode config
# NNODES=${WORLD_SIZE}
# NODE_RANK=${RANK}
# GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# # GPUS_PER_NODE=1
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# echo $DISTRIBUTED_ARGS

# change LOAD to your local path of DocOwl1.5-stage1
LOAD='mPLUG/DocOwl1.5-Omni'

OUTPUT_DIR=/scratch/slurm-user23-kaist/son/code/mPLUG-DocOwl/DocOwl1.5/checkpoints/docowl1.5-omni-instrucdoc-lora-1e4-azure

# batch size = per_device_train_batch_size x GPUS_PER_NODE x NNODES x gradient_accumulation_steps
DATA_FILE=/scratch/slurm-user23-kaist/son/code/VD-Instruct/data/vd-instruct-2/train/azure/instructdoc_docowl_train_5k.jsonl

# CUDA_VISIBLE_DEVICES=3 torchrun mplug_docowl/train/train_docowl.py \
deepspeed /scratch/slurm-user23-kaist/son/code/mPLUG-DocOwl/DocOwl1.5/mplug_docowl/train/train_docowl.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --vision2text_lr 2e-5 \
    --deepspeed /scratch/slurm-user23-kaist/son/code/mPLUG-DocOwl/DocOwl1.5/scripts/zero2.json \
    --model_name_or_path $LOAD \
    --cache_dir /scratch/slurm-user23-kaist/son/code/mPLUG-DocOwl/DocOwl1.5/checkpoints \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder /scratch/slurm-user23-kaist/son/data/instructdocs/raw_datasets \
    --image_size 448 \
    --crop_anchors 'grid_9' \
    --add_global_img True \
    --add_textual_crop_indicator True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3600 \
    --gradient_checkpointing True \
    --tune_vision2text True \
    --freeze_vision_model True \
    --freeze_backbone True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb