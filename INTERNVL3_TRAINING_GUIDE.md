# InternVL3 Full Parameter Finetuning Guide

## Prerequisites

1. **Install LLaMA-Factory dependencies:**
```bash
cd /home/yu/Downloads/finetune_data/LLaMA-Factory
pip install -e ".[torch,metrics]"
```

2. **GPU Requirements:**
   - **InternVL3-1B**: 1x A100 (40GB) or 1x H100
   - **InternVL3-2B**: 1x A100 (80GB) or 2x A100 (40GB)
   - **InternVL3-4B**: 2x A100 (80GB) or 4x A100 (40GB)
   - **InternVL3-8B**: 4x A100 (80GB) or 8x A100 (40GB)

## Quick Start

### Method 1: Using the Shell Script (Recommended)

```bash
cd /home/yu/Downloads/finetune_data/LLaMA-Factory

# Make script executable
chmod +x train_internvl3_pointing.sh

# Edit the script to set your model variant
# MODEL_NAME options:
#   - OpenGVLab/InternVL3-1B
#   - OpenGVLab/InternVL3-2B
#   - OpenGVLab/InternVL3-4B
#   - OpenGVLab/InternVL3-8B
nano train_internvl3_pointing.sh

# Run training
./train_internvl3_pointing.sh
```

### Method 2: Direct Python Command

```bash
cd /home/yu/Downloads/finetune_data/LLaMA-Factory

python src/train.py \
    --stage sft \
    --do_train True \
    --model_name_or_path OpenGVLab/InternVL3-8B \
    --dataset pointing_combined_train \
    --template intern_vl \
    --finetuning_type full \
    --output_dir ./output/internvl3_pointing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --bf16 True \
    --save_steps 500 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --report_to tensorboard
```

## Configuration Options

### Model Variants

| Model | Parameters | VRAM (Full) | Recommended GPUs |
|-------|-----------|-------------|------------------|
| InternVL3-1B | 1B | ~12GB | 1x A100 40GB |
| InternVL3-2B | 2B | ~24GB | 1x A100 80GB |
| InternVL3-4B | 4B | ~48GB | 2x A100 80GB |
| InternVL3-8B | 8B | ~96GB | 4x A100 80GB |

### Training Hyperparameters

**Conservative (Recommended for Full Parameter Training):**
```bash
--learning_rate 1e-5 \
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4
```

**Aggressive (If you have experience):**
```bash
--learning_rate 2e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2
```

### Memory Optimization

If you run into OOM (Out of Memory) errors:

**1. Reduce Batch Size:**
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8
```

**2. Enable Gradient Checkpointing:**
```bash
--gradient_checkpointing True
```

**3. Use DeepSpeed ZeRO Stage 2:**
```bash
--deepspeed configs/ds_z2_config.json
```

**4. Use DeepSpeed ZeRO Stage 3 (for very large models):**
```bash
--deepspeed configs/ds_z3_config.json
```

## Training with DeepSpeed (Multi-GPU)

### DeepSpeed ZeRO Stage 2 Configuration

Create `ds_z2_config.json`:
```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 2
}
```

Run with:
```bash
deepspeed --num_gpus=4 src/train.py \
    --deepspeed ds_z2_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path OpenGVLab/InternVL3-8B \
    --dataset pointing_combined_train \
    --template intern_vl \
    --finetuning_type full \
    --output_dir ./output/internvl3_pointing \
    --bf16 True \
    --num_train_epochs 3
```

## Training Process

### 1. Start Training
```bash
cd /home/yu/Downloads/finetune_data/LLaMA-Factory
./train_internvl3_pointing.sh
```

### 2. Monitor Training
```bash
# In another terminal
tensorboard --logdir ./output/internvl3_pointing
# Open http://localhost:6006 in browser
```

### 3. Check Training Logs
```bash
tail -f ./output/internvl3_pointing/trainer_log.jsonl
```

## Evaluation

After training, evaluate on the test set:

```bash
python src/train.py \
    --stage sft \
    --do_predict True \
    --model_name_or_path ./output/internvl3_pointing \
    --dataset pointing_combined_test \
    --template intern_vl \
    --finetuning_type full \
    --output_dir ./output/internvl3_pointing_eval \
    --per_device_eval_batch_size 4 \
    --predict_with_generate True
```

## Testing the Model

Create a test script `test_pointing.py`:

```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch

# Load model
model_path = "./output/internvl3_pointing"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Test image
image = Image.open("../final_pointing/general_object_pointing_image/0_original.jpg")

# Question
question = """The question is: Identify the person in the image.
Please directly output the pixel location of the target point.
Format your answer as (x, y), where:
- x is the horizontal coordinate (left → right),
- y is the vertical coordinate (top → bottom).
Both x and y must be normalized to the range [0, 1] as floating-point values, indicating the relative position of the point within the image."""

# Generate
response = model.chat(tokenizer, image, question, generation_config=dict(max_new_tokens=100))
print(f"Model prediction: {response}")
```

## Training Tips

### 1. Start Small
- Train on a subset first to verify everything works
- Use smaller model variant (1B or 2B) for initial experiments

### 2. Learning Rate
- Full parameter training: `1e-5` to `2e-5`
- If loss doesn't decrease: try `5e-6`
- If loss explodes: lower to `5e-6` or `1e-6`

### 3. Epochs
- Start with 3 epochs
- Monitor validation loss to avoid overfitting
- For 516 samples, 3-5 epochs is usually sufficient

### 4. Batch Size
- Larger effective batch size = more stable training
- `per_device_batch_size × gradient_accumulation_steps × num_gpus`
- Aim for effective batch size of 8-16

### 5. Save Checkpoints
```bash
--save_steps 100 \
--save_total_limit 3 \
--save_only_model True
```

## Troubleshooting

### Issue: CUDA Out of Memory
**Solutions:**
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing`
4. Use DeepSpeed ZeRO

### Issue: Loss Not Decreasing
**Solutions:**
1. Lower learning rate to `5e-6`
2. Increase warmup steps to `200`
3. Check if data is loading correctly
4. Verify image paths are accessible

### Issue: Training Too Slow
**Solutions:**
1. Use `bf16=True` (faster than fp16 on A100/H100)
2. Increase batch size if memory allows
3. Use multiple GPUs with DeepSpeed
4. Enable `tf32=True` on Ampere GPUs

### Issue: Model Outputs Gibberish
**Solutions:**
1. Check if template is correct (`--template intern_vl`)
2. Verify dataset format is correct
3. Lower learning rate
4. Check if model loaded correctly

### Issue: Template Error
If you get a template error, try:
```bash
--template default  # or internvl
```
Check available templates:
```bash
python -c "from llamafactory.data import get_template_and_fix_tokenizer; print(get_template_and_fix_tokenizer.__doc__)"
```

## Expected Results

After training, your model should:
- Output coordinates in format `(x, y)`
- Coordinates between 0.0 and 1.0
- Point to objects mentioned in questions
- Handle both general objects and spatial relationships

## Advanced: Mixed Precision Training

**FP16 (older GPUs):**
```bash
--fp16 True
```

**BF16 (A100/H100, recommended):**
```bash
--bf16 True
```

**TF32 (automatic on Ampere):**
```bash
--tf32 True
```

## Training Timeline Estimates

| Model | GPUs | Batch Size | Time/Epoch | Total Time (3 epochs) |
|-------|------|------------|------------|---------------------|
| 1B | 1x A100 | 4 | ~15 min | ~45 min |
| 2B | 1x A100 | 4 | ~25 min | ~75 min |
| 4B | 2x A100 | 8 | ~30 min | ~90 min |
| 8B | 4x A100 | 8 | ~40 min | ~2 hours |

*Based on 516 training samples

## Dataset Information

- **Training samples**: 516
- **Test samples**: 130
- **Format**: ShareGPT with images
- **Image paths**: Relative (`../final_pointing/...`)
- **Task**: Point prediction (normalized coordinates)
- **Ground truth**: Truly centered points (sampled from closest 20% to centroid)

## Key Differences from InternVL2.5

1. **Template**: Use `--template intern_vl` (not `intern2`)
2. **Model names**: `OpenGVLab/InternVL3-*` (not `InternVL2_5-*`)
3. **Architecture**: InternVL3 has improved vision-language alignment
4. **Performance**: Better multimodal understanding

## Next Steps

1. ✅ Dataset is ready
2. ⏭️ **Run training**: `./train_internvl3_pointing.sh`
3. ⏭️ **Monitor with tensorboard**
4. ⏭️ **Evaluate on test set**
5. ⏭️ **Test with custom images**

Good luck with your training! 🚀
