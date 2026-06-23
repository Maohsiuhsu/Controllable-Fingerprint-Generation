#!/usr/bin/env python
# coding=utf-8
import argparse
import time
import logging
import math
import os
import gc

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from itertools import chain
from torch.utils.data import TensorDataset, DataLoader

from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from prompt_learner import PromptLearner

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Single-Concept Prompt Distribution Learning with LoRA.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--n_ctx", type=int, default=8)
    parser.add_argument("--n_prompts", type=int, default=4)
    parser.add_argument("--customize_prefix", type=str, default="")
    parser.add_argument("--customize_suffix", type=str, default="")
    parser.add_argument("--reparam_samples", type=int, default=4)
    parser.add_argument("--ortho_loss_weight", type=float, default=0.001)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--validation_steps", type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None: set_seed(args.seed)
    if accelerator.is_main_process: os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # =======================================================
    # 1. 載入 SD Pipeline 並掛載 LoRA
    # =======================================================
    logger.info(f"[LoRA] Loading base SD and LoRA from: {args.lora_path} (Scale: {args.lora_scale})")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype, safety_checker=None
    )
    base_pipe.load_lora_weights(args.lora_path, adapter_name="default")
    base_pipe.set_adapters(["default"], adapter_weights=[args.lora_scale])

    vae = base_pipe.vae.to(accelerator.device, dtype=weight_dtype)
    unet = base_pipe.unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder = base_pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    tokenizer = base_pipe.tokenizer

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # =======================================================
    # 2. 資料集預處理 (無增強，準備 Cache)
    # =======================================================
    dataset = load_dataset("imagefolder", data_files={"train": os.path.join(args.train_data_dir, "**")})
    train_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        return {"pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float()}
    
    temp_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    # =======================================================
    # 3. Latent RAM Cache 機制 (極限加速與省顯存)
    # =======================================================
    logger.info("***** Caching Latents into RAM *****")
    vae.eval()
    all_latents = []
    with torch.no_grad():
        for batch in tqdm(temp_dataloader, desc="Encoding Images"):
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.mode() * vae.config.scaling_factor
            all_latents.append(latents.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    logger.info(f"Cached {all_latents.shape[0]} latents. Unloading VAE to free VRAM!")
    
    # 釋放 VAE 與多餘的 Pipeline 元件
    del vae
    del base_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 建立全新的輕量化 DataLoader
    ram_dataset = TensorDataset(all_latents)
    train_dataloader = DataLoader(ram_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)

    # =======================================================
    # 4. 初始化 Prompt Learner (傳入帶有 LoRA 的 TE)
    # =======================================================
    prompt_learner = PromptLearner(
        tokenizer=tokenizer,
        text_encoder_instance=text_encoder,
        n_ctx=args.n_ctx,
        n_prompts=args.n_prompts,
        dtype=weight_dtype,
        customize_prefix=args.customize_prefix,
        customize_suffix=args.customize_suffix,
        reparam_samples=args.reparam_samples,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(
        prompt_learner.parameters(), lr=args.learning_rate, weight_decay=1e-2, eps=1e-08
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, 
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, prompt_learner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, prompt_learner, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("DreamDistribution_SingleConcept", config=vars(args))

    def compute_snr(timesteps):
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape): sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape): sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
        return (alpha / sigma) ** 2

    # =======================================================
    # 5. 訓練迴圈
    # =======================================================
    logger.info("***** Running training *****")
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        prompt_learner.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(prompt_learner):
                # 直接讀取記憶體中的 Latents
                latents = batch[0].to(accelerator.device, dtype=weight_dtype)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Prompt Learner 推論 (不需傳 input_ids，內部自動處理單一概念)
                dummy_idx = torch.zeros(bsz, dtype=torch.long, device=accelerator.device)
                with accelerator.autocast():
                    prompt_hidden_states, ortho_loss = prompt_learner(cls_idx=dummy_idx)

                # 準備 UNet 輸入 (擴充 batch size 對應 reparam_samples)
                with accelerator.autocast():
                    prompt_hidden_states = rearrange(prompt_hidden_states, "b n l d -> (b n) l d")
                    noisy_latents_rep = repeat(noisy_latents, "b c h w -> (b n) c h w", n=args.reparam_samples)
                    timesteps_rep = repeat(timesteps, "t -> (t n)", n=args.reparam_samples)
                    model_pred = unet(noisy_latents_rep, timesteps_rep, prompt_hidden_states).sample
                    model_pred = rearrange(model_pred, "(b n) c h w -> b n c h w", b=bsz, n=args.reparam_samples)

                target = noise
                target = repeat(target, "b c h w -> b n c h w", n=args.reparam_samples)
                
                # SNR 加權損失
                snr = compute_snr(timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                mse_loss_weights = repeat(mse_loss_weights, "b -> b n", n=args.reparam_samples)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(2, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
                
                # 加上正交損失
                loss += args.ortho_loss_weight * ortho_loss
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prompt_learner.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "ortho_loss": ortho_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save(accelerator.unwrap_model(prompt_learner).state_dict(), save_path)

            progress_bar.set_postfix({"loss": loss.detach().item()})
            if global_step >= args.max_train_steps: break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(prompt_learner)
        save_path = os.path.join(args.output_dir, "final_prompt_learner.pt")
        npz_save_path = os.path.join(args.output_dir, "final_prompt_learner.npz")
        
        torch.save(unwrapped.state_dict(), save_path)
        
        unwrapped.eval()
        with torch.no_grad(), accelerator.autocast():
            unwrapped.fit(prefix=args.customize_prefix, suffix=args.customize_suffix)
            
        np.savez(
            npz_save_path,
            means=unwrapped.means,
            stds=unwrapped.stds,
            prompt_texts=unwrapped.prompt_texts,
            all_prompts=unwrapped.ctx.detach().cpu().float().numpy()
        )
        logger.info(f"✅ Training Complete! Saved to {npz_save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()