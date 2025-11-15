#!/bin/bash
# Full Parameter Finetuning Script for InternVL3 on Pointing Dataset
# This script performs full parameter training (not LoRA)

# Configuration
MODEL_NAME="OpenGVLab/InternVL3-8B"  # Change to your InternVL3 variant
DATASET="pointing_combined_train"
OUTPUT_DIR="./output/internvl3_pointing_full"
NUM_EPOCHS=3
BATCH_SIZE=2  # Adjust based on your GPU memory
GRAD_ACCUM_STEPS=4  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE=1e-5  # Lower LR for full parameter training
MAX_LENGTH=2048

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training
python src/train.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${MODEL_NAME} \
    --dataset ${DATASET} \
    --template intern_vl \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache True \
    --overwrite_output_dir True \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --ddp_timeout 9000 \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --cutoff_len ${MAX_LENGTH} \
    --save_steps 500 \
    --plot_loss True \
    --num_train_epochs ${NUM_EPOCHS} \
    --bf16 True \
    --save_only_model True \
    --report_to tensorboard

echo "Training complete! Model saved to: ${OUTPUT_DIR}"
