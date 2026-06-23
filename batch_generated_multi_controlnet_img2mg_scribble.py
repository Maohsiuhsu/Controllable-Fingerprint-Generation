#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import os
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.random import normal, seed as np_seed
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

from prompt_learner import PromptLearner


torch.backends.cuda.matmul.allow_tf32 = True

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# Args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(
        "Img2Img + Scribble + Shape + PromptLearner fingerprint generation"
    )

    p.add_argument("--source_dir", type=str, required=True)
    p.add_argument("--style_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--learner_weights", type=str, required=True)

    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )

    p.add_argument("--lora_path", type=str, default="")
    p.add_argument("--lora_scale", type=float, default=0.8)

    p.add_argument("--customize_prefix", type=str, default="")
    p.add_argument("--customize_suffix", type=str, default="touch-free roll fingerprint")

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_source_images", type=int, default=0)

    # 強制 batch_size=1；保留參數只是為了相容你的 bash script
    p.add_argument("--batch_size", type=int, default=1)

    p.add_argument("--inference_steps", type=int, default=35)
    p.add_argument("--guidance_scale", type=float, default=6.5)
    p.add_argument("--strength", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)

    # Scribble map 仍然可以用 Canny 取邊線後 dilate 成 scribble
    p.add_argument("--scribble_low", type=int, default=80)
    p.add_argument("--scribble_high", type=int, default=160)
    p.add_argument("--scribble_scale", type=float, default=1.0)

    p.add_argument("--shape_scale", type=float, default=0.8)

    p.add_argument("--scribble_end", type=float, default=1.0)
    p.add_argument("--shape_end", type=float, default=1.0)

    p.add_argument("--num_visualize", type=int, default=5)
    p.add_argument(
        "--visualize_mode",
        type=str,
        default="random",
        choices=["random", "head"],
    )

    p.add_argument("--save_controls", action="store_true")

    return p.parse_args()


# =========================================================
# File utils
# =========================================================
def list_images(root: str):
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def resize_and_pad(pil_img: Image.Image, res: int) -> Image.Image:
    w, h = pil_img.size
    scale = float(res) / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (res, res), color=(255, 255, 255))

    x = (res - new_w) // 2
    y = (res - new_h) // 2
    canvas.paste(resized, (x, y))

    return canvas


# =========================================================
# Control maps
# =========================================================
def get_fingerprint_mask(pil_img: Image.Image) -> np.ndarray:
    gray = np.array(pil_img.convert("L"))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        240,
        255,
        cv2.THRESH_BINARY_INV,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8),
        iterations=2,
    )

    return mask


def get_scribble_map(
    src_img: Image.Image,
    low: int,
    high: int,
    use_mask: bool = True,
) -> Image.Image:
    """
    Scribble ControlNet 通常接受黑底白線。
    這裡用 Canny 抽 ridge/edge，再 dilate 讓線條更像 scribble。
    """
    gray = np.array(src_img.convert("L"))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edge = cv2.Canny(blur, low, high)

    if use_mask:
        mask = get_fingerprint_mask(src_img)
        edge = cv2.bitwise_and(edge, mask)

    # 讓 scribble 線條稍微加粗
    edge = cv2.dilate(
        edge,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )

    return Image.fromarray(edge).convert("L")


def get_shape_map(style_img: Image.Image) -> Image.Image:
    mask = get_fingerprint_mask(style_img)
    return Image.fromarray(mask).convert("L")


# =========================================================
# Prompt Learner
# =========================================================
def load_learner(weights_path, args, pipe, device="cuda"):
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    new_ckpt = OrderedDict(
        {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in state_dict.items()
            if "ctx" in k
        }
    )

    if "ctx" not in new_ckpt:
        raise RuntimeError("Prompt Learner checkpoint 中找不到 ctx。")

    learner = PromptLearner(
        pipe.tokenizer,
        pipe.text_encoder,
        new_ckpt["ctx"].shape[2],
        new_ckpt["ctx"].shape[1],
        pipe.text_encoder.dtype,
        args.customize_prefix,
        args.customize_suffix,
    ).to(device)

    learner.load_state_dict(new_ckpt, strict=False)
    learner.eval()

    with torch.no_grad():
        learner.fit(
            prefix=args.customize_prefix,
            suffix=args.customize_suffix,
        )

    return learner.means, learner.stds


def sample_prompt_embedding(means, stds, seed_value, device="cuda"):
    np_seed(seed_value)

    epsilon = normal(
        loc=0,
        scale=1,
        size=means[0].shape,
    ).astype(np.float32)

    alpha = 0.2

    emb = torch.from_numpy(
        means[0] + alpha * epsilon * stds[0]
    ).unsqueeze(0).to(device, dtype=torch.float16)

    return emb


# =========================================================
# Visualization
# =========================================================
def add_title(img: Image.Image, title: str) -> Image.Image:
    from PIL import ImageDraw

    img = img.convert("RGB")
    w, h = img.size

    canvas = Image.new("RGB", (w, h + 36), (230, 230, 230))
    canvas.paste(img, (0, 36))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill=(255, 0, 0))

    return canvas


def make_panel(images, titles, save_path):
    images = [add_title(im, t) for im, t in zip(images, titles)]

    w = sum(im.size[0] for im in images)
    h = max(im.size[1] for im in images)

    panel = Image.new("RGB", (w, h), (240, 240, 240))

    x = 0
    for im in images:
        panel.paste(im, (x, 0))
        x += im.size[0]

    panel.save(save_path)


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    if args.batch_size != 1:
        print(
            "[Warning] Img2Img + Multi-ControlNet 在目前 diffusers 版本不支援 batch_size > 1。"
            "已強制改成 batch_size=1。"
        )
        args.batch_size = 1

    device = "cuda"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    out_img_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(out_img_dir, exist_ok=True)

    vis_dir = os.path.join(args.output_dir, "visualized")
    os.makedirs(vis_dir, exist_ok=True)

    ctrl_dir = os.path.join(args.output_dir, "controls")
    if args.save_controls:
        os.makedirs(ctrl_dir, exist_ok=True)

    # -----------------------------------------------------
    # 1) Build source/style list
    # -----------------------------------------------------
    source_list = list(list_images(args.source_dir))
    style_list = list(list_images(args.style_dir))

    if len(source_list) == 0:
        raise RuntimeError("No source images found.")

    if len(style_list) == 0:
        raise RuntimeError("No style images found.")

    if args.max_source_images > 0:
        source_list = source_list[:args.max_source_images]

    total = len(source_list)

    if args.visualize_mode == "head":
        vis_indices = set(range(min(args.num_visualize, total)))
    else:
        vis_indices = set(
            random.sample(
                range(total),
                min(args.num_visualize, total),
            )
        )

    print(f"[Info] Source images: {len(source_list)}")
    print(f"[Info] Style images: {len(style_list)}")
    print(f"[Info] Visualized indices: {sorted(vis_indices)}")
    print(f"[Info] strength={args.strength}")
    print(f"[Info] batch_size forced to 1")

    # -----------------------------------------------------
    # 2) Load model
    # -----------------------------------------------------
    print("[1/4] Loading ControlNets...")

    cn_scribble = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float16,
    )

    cn_shape = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_softedge",
        torch_dtype=torch.float16,
    )

    print("[2/4] Loading Img2Img ControlNet pipeline...")

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=[cn_scribble, cn_shape],
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )

    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[Info] xFormers enabled.")
    except Exception:
        print("[Info] xFormers not available, skip.")

    try:
        pipe.unet.to(memory_format=torch.channels_last)
        print("[Info] channels_last enabled.")
    except Exception:
        pass

    # -----------------------------------------------------
    # 3) LoRA / Prompt Learner
    # -----------------------------------------------------
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"[3/4] Loading LoRA: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path, adapter_name="fp")
        pipe.set_adapters(["fp"], adapter_weights=[args.lora_scale])
    else:
        print("[3/4] No LoRA loaded.")

    print("[4/4] Loading Prompt Learner...")

    means, stds = load_learner(
        args.learner_weights,
        args,
        pipe,
        device=device,
    )

    # -----------------------------------------------------
    # 4) Prepare jobs
    # -----------------------------------------------------
    print("[Prepare] Building jobs...")

    jobs = []
    shape_cache = {}

    for idx, src_path in enumerate(tqdm(source_list, desc="Preparing")):
        try:
            src_pil = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"[Skip] Cannot open source: {src_path}, {repr(e)}")
            continue

        style_path = style_list[idx % len(style_list)]

        try:
            style_pil = Image.open(style_path).convert("RGB")
        except Exception as e:
            print(f"[Skip] Cannot open style: {style_path}, {repr(e)}")
            continue

        src_padded = resize_and_pad(src_pil, args.resolution)

        if style_path not in shape_cache:
            style_padded = resize_and_pad(style_pil, args.resolution)
            shape_map = get_shape_map(style_padded)

            shape_cache[style_path] = {
                "style_padded": style_padded,
                "shape_map": shape_map,
            }
        else:
            style_padded = shape_cache[style_path]["style_padded"]
            shape_map = shape_cache[style_path]["shape_map"]

        scribble_map = get_scribble_map(
            src_padded,
            low=args.scribble_low,
            high=args.scribble_high,
            use_mask=True,
        )

        prompt_emb = sample_prompt_embedding(
            means,
            stds,
            args.seed + idx,
            device=device,
        )

        out_name = f"{Path(src_path).stem}.png"
        out_path = os.path.join(out_img_dir, out_name)

        visualize = idx in vis_indices

        jobs.append(
            {
                "idx": idx,
                "src_path": str(src_path),
                "style_path": str(style_path),
                "init_image": src_padded,
                "src_padded": src_padded if visualize else None,
                "style_padded": style_padded if visualize else None,
                "scribble_map": scribble_map,
                "shape_map": shape_map,
                "prompt_emb": prompt_emb,
                "seed": args.seed + idx,
                "out_path": out_path,
                "out_name": out_name,
                "visualize": visualize,
            }
        )

        if args.save_controls:
            stem = Path(src_path).stem
            scribble_map.save(os.path.join(ctrl_dir, f"{stem}_scribble.png"))
            shape_map.save(os.path.join(ctrl_dir, f"{stem}_shape.png"))

    if len(jobs) == 0:
        raise RuntimeError("No valid jobs prepared.")

    print(f"[Info] Total jobs: {len(jobs)}")
    print(f"[Info] Cached shape maps: {len(shape_cache)}")

    # -----------------------------------------------------
    # 5) Img2Img inference, one image at a time
    # -----------------------------------------------------
    print("[Run] Start img2img inference with batch_size=1...")

    with torch.inference_mode():
        for img_idx, job in enumerate(jobs, start=1):
            print(f"[Image {img_idx}/{len(jobs)}] {job['out_name']}")

            generator = torch.Generator(device=device).manual_seed(job["seed"])

            images = pipe(
                prompt_embeds=job["prompt_emb"],
                image=job["init_image"],
                control_image=[
                    job["scribble_map"],
                    job["shape_map"],
                ],
                strength=args.strength,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=[
                    args.scribble_scale,
                    args.shape_scale,
                ],
                control_guidance_end=[
                    args.scribble_end,
                    args.shape_end,
                ],
                generator=generator,
            ).images

            img = images[0]
            img_gray = img.convert("L")
            img_gray.save(job["out_path"])

            if job["visualize"]:
                sample_dir = os.path.join(
                    vis_dir,
                    Path(job["out_name"]).stem,
                )
                os.makedirs(sample_dir, exist_ok=True)

                img_rgb = img.convert("RGB")

                job["src_padded"].save(os.path.join(sample_dir, "01_source.png"))
                job["style_padded"].save(os.path.join(sample_dir, "02_style.png"))
                job["scribble_map"].save(os.path.join(sample_dir, "03_scribble.png"))
                job["shape_map"].save(os.path.join(sample_dir, "04_shape.png"))
                img_rgb.save(os.path.join(sample_dir, "05_generated_rgb.png"))
                img_gray.save(os.path.join(sample_dir, "06_generated_gray.png"))

                make_panel(
                    images=[
                        job["src_padded"],
                        job["style_padded"],
                        job["scribble_map"].convert("RGB"),
                        job["shape_map"].convert("RGB"),
                        img_rgb,
                        Image.merge("RGB", (img_gray, img_gray, img_gray)),
                    ],
                    titles=[
                        "Source",
                        "Style",
                        "Scribble",
                        "Shape",
                        "Generated RGB",
                        "Generated Gray",
                    ],
                    save_path=os.path.join(sample_dir, "panel.png"),
                )

            del images
            torch.cuda.empty_cache()
            gc.collect()

    print("\n[DONE]")
    print(f"Generated images: {out_img_dir}")
    print(f"Visualized samples: {vis_dir}")

    if args.save_controls:
        print(f"Controls: {ctrl_dir}")


if __name__ == "__main__":
    main()