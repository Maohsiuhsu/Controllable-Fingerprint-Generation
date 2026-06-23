# Fingerprint Prompt Learner

This project trains a Prompt Learner and uses it to generate fingerprint images with Stable Diffusion, LoRA, and ControlNet.

## 1. Install Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate datasets einops opencv-python pillow tqdm numpy tensorboard
```

Optional:

```bash
pip install xformers
```

## 2. Prepare Data

Recommended folder structure:

```text
project/
├── train_with_lora_v2.py
├── prompt_learner.py
├── batch_generated_multi_controlnet_img2mg_scribble.py
├── data/
│   ├── train/      # images for training Prompt Learner
│   ├── source/     # source fingerprint images for generation
│   └── style/      # style / shape reference images
├── lora/           # LoRA weights
└── outputs/
```

## 3. Train Prompt Learner

```bash
accelerate launch train_with_lora_v2.py \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --train_data_dir ./data/train \
  --output_dir ./outputs/prompt_learner \
  --lora_path ./lora \
  --lora_scale 1.0 \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 50000 \
  --learning_rate 1e-3 \
  --mixed_precision bf16 \
  --n_ctx 8 \
  --n_prompts 4 \
  --reparam_samples 4 \
  --customize_suffix "touch-free roll fingerprint"
```

After training, the main output will be:

```text
./outputs/prompt_learner/final_prompt_learner.pt
```

## 4. Generate Images

```bash
python batch_generated_multi_controlnet_img2mg_scribble.py \
  --source_dir ./data/source \
  --style_dir ./data/style \
  --output_dir ./outputs/generated_samples \
  --learner_weights ./outputs/prompt_learner/final_prompt_learner.pt \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --lora_path ./lora \
  --lora_scale 0.8 \
  --customize_suffix "touch-free roll fingerprint" \
  --resolution 512 \
  --inference_steps 35 \
  --guidance_scale 6.5 \
  --strength 0.9 \
  --scribble_scale 1.0 \
  --shape_scale 0.8 \
  --num_visualize 5 \
  --save_controls
```

Generated results will be saved to:

```text
./outputs/generated_samples/generated/
```

Visualization results will be saved to:

```text
./outputs/generated_samples/visualized/
```

## Notes

- Use the same base model, LoRA, resolution, and prompt suffix for training and generation.
- `final_prompt_learner.pt` is required for generation.
- The generation script requires CUDA.
