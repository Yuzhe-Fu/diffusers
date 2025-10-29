# eval_sdm_coco_ditto.py
# SD v3.5 (DiT) or other text-to-image pipelines on local MSCOCO val2017.
# Computes FID/IS/CLIP-Score, saves a subset of generated images and an optional grid.

'''
# baseline (FP16 everywhere)
CUDA_VISIBLE_DEVICES=3 python eval_sdm_coco_ditto.py --mscoco_root /home/zs89/EchoFlow/mscoco --model stabilityai/stable-diffusion-3.5-medium --num_samples 100 --steps 50 --batch 8 --precision fp16 --height 1024 --width 1024 --fid_ref full --save_dir sdm35_mscoco_baseline --save_k 20 --save_grid --ditto_off

# Ditto W8A8 fake Dynamics --> probably explode due to memory (need to be fixed)
CUDA_VISIBLE_DEVICES=3 python eval_sdm_coco_ditto.py --mscoco_root /home/zs89/EchoFlow/mscoco --model stabilityai/stable-diffusion-3.5-medium --num_samples 100 --steps 50 --batch 8 --precision fp16 --height 1024 --width 1024 --fid_ref full --save_dir sdm35_mscoco_ditto --save_k 20 --save_grid

# W8A8 (fake dynamics, no ditto)
CUDA_VISIBLE_DEVICES=3 python eval_sdm_coco_ditto.py --mscoco_root /home/zs89/EchoFlow/mscoco --model stabilityai/stable-diffusion-3.5-medium --num_samples 100 --steps 50 --batch 8 --precision fp16 --height 1024 --width 1024 --fid_ref full --save_dir sdm35_coco_w8a8 --save_k 20 --save_grid --w8a8_only

'''

import argparse, os, sys, math
from typing import List
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image

from ditto import DittoConfig, attach_ditto
from mscoco_local import MSCOCOVal2017
from metrics_eval import (
    InceptionWrap, compute_moments, frechet_distance,
    inception_score_from_logits, CLIPScorer
)
#from old.qdiffusion import QDCalibrator

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def pick_dtype(precision: str, has_cuda: bool):
    precision = precision.lower()
    if precision == "fp16":
        return torch.float16 if has_cuda else torch.float32
    if precision == "bf16":
        return torch.bfloat16 if has_cuda else torch.float32
    return torch.float32  # fp32 default

def save_image_grid(images: List[Image.Image], nrow: int, out_path: str):
    if len(images) == 0:
        return
    nrow = max(1, nrow)
    ncol = math.ceil(len(images) / nrow)
    w, h = images[0].size
    grid = Image.new("RGB", (nrow * w, ncol * h))
    for idx, im in enumerate(images):
        r = idx // nrow
        c = idx % nrow
        grid.paste(im, (c * w, r * h))
    grid.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    # Local MSCOCO root
    ap.add_argument("--mscoco_root", type=str, required=True,
                    help="Path to MSCOCO root containing 'val2017/' and 'annotations/'")
    ap.add_argument("--num_samples", type=int, default=5000, help="How many images to evaluate (<=5000)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pick", type=str, default="random", choices=["random", "first"],
                    help="Which caption to use when multiple exist")

    # FID reference: full val5k or subset
    ap.add_argument("--fid_ref", type=str, default="full", choices=["full", "subset"],
                    help="Use full val2017 for real stats (recommended) or only subset size.")
    ap.add_argument("--cache_real_stats", type=str, default="mscoco_val2017_inception_moments.npz")
    ap.add_argument("--recompute_real", action="store_true",
                    help="Ignore cache and recompute real COCO moments this run.")

    # SDM / sampler
    ap.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium",
                    help="HF model id for AutoPipelineForText2Image")
    ap.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="Pipeline dtype. Paper reports FID at FP32.")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--ditto_off", action="store_true", help="Disable Ditto/quantization; run pure baseline")
    ap.add_argument("--w8a8_only", action="store_true", help="Enable W8A8 fake_dynamic quantization only (no Δ path)")
    # Ditto controls
    ap.add_argument("--softmax_offload", action="store_true",
                    help="Compute attention softmax on CPU and stream P@V back in chunks (Ditto)")
    ap.add_argument("--softmax_chunk", type=int, default=2048,
                    help="Columns per chunk when --softmax_offload is used")

    # Saving
    ap.add_argument("--save_dir", type=str, default="sdm_mscoco_samples",
                    help="Directory to save generated images")
    ap.add_argument("--save_k", type=int, default=64,
                    help="Save first K generated images (to avoid writing 5k files)")
    ap.add_argument("--save_grid", action="store_true",
                    help="Also save a grid preview (first min(save_k, 16) images).")

    # CLIP backbone and IS splits
    ap.add_argument("--clip_backbone", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--is_splits", type=int, default=10, help="Upper bound on IS splits; clamped to N.")

    # Sanity option
    ap.add_argument("--sanity", action="store_true", help="Compute FID(gen vs gen) ~= 0 as a sanity check.")

    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(args.precision, device.type == "cuda")

    # No calibration JSON path; Ditto runs in fake_dynamic W8A8 by default now.

    if args.num_samples < 2000:
        print(f"[warn] num_samples={args.num_samples} is small — FID will be high/unstable. "
              f"Use the full val set (5000) for stable estimates.", file=sys.stderr)

    # -------- MSCOCO local loader --------
    loader = MSCOCOVal2017(
        root=args.mscoco_root,
        num_samples=args.num_samples,
        seed=args.seed,
        pick=args.pick,
    )
    gen_img_paths, prompts, all_val_paths = loader.load()
    if len(gen_img_paths) == 0:
        raise RuntimeError("No MSCOCO images found. Check --mscoco_root path.")
    print(f"[Data/MSCOCO] Using {len(gen_img_paths)} prompts from val2017. Total val images: {len(all_val_paths)}")

    # -------- Build SD pipeline (AutoPipeline) --------
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass
    print(f"[Pipeline] Loaded {args.model}; steps = {args.steps} | dtype = {dtype}")

    # -------- Attach Ditto (algorithmic + quant) --------
    if args.ditto_off:
        mode_desc = "baseline"
        print("[Mode] Baseline (no Ditto, no quantization).")
    elif args.w8a8_only:
        cfg = DittoConfig(
            enable=True,
            quant_mode="fake_dynamic",
            a_bits=8, w_bits=8, per_channel_w=True,
            qd_scales=None, qd_time_bins=1,
            total_steps=args.steps,
            enable_defo=False,
            enable_attention_ditto=bool(args.softmax_offload),  # only install processors if offloading
            print_layer_choices=False,
            quantize_attn_only=True,
            quantize_to_q_only=False,
            force_orig_all=True,  # disable Δ path; pure W8A8
            delta_int4=False,
            zero_skip=False,
            softmax_offload=args.softmax_offload,
            softmax_chunk=args.softmax_chunk,
        )
        attach_ditto(pipe, cfg)
        mode_desc = f"W8A8 fake_dynamic (quant-only, softmax_offload={args.softmax_offload})"
        print(f"[Mode] {mode_desc}")
    else:
        cfg = DittoConfig(
            enable=True,
            quant_mode="fake_dynamic",
            a_bits=8, w_bits=8, per_channel_w=True,
            qd_scales=None, qd_time_bins=1,
            total_steps=args.steps,
            enable_defo=True, enable_attention_ditto=True, print_layer_choices=False,
            quantize_attn_only=True,
            quantize_to_q_only=False,
            force_orig_all=False, delta_int4=True, zero_skip=True,
            softmax_offload=args.softmax_offload, softmax_chunk=args.softmax_chunk,
        )
        attach_ditto(pipe, cfg)
        mode_desc = f"Ditto W8A8 (softmax_offload={args.softmax_offload})"
        print(f"[Mode] {mode_desc}")

    # -------- Generate images --------
    os.makedirs(args.save_dir, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    gen_images: List[Image.Image] = []
    print("[Gen] Generating images with Stable Diffusion...")
    for i in range(0, len(prompts), args.batch):
        batch_prompts = prompts[i: i + args.batch]
        with torch.inference_mode():
            out = pipe(
                batch_prompts,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                width=args.width, height=args.height,
            )
        imgs = [im.convert("RGB") for im in out.images]
        gen_images.extend(imgs)

    # Save first K generated images + optional grid
    k = min(args.save_k, len(gen_images))
    for idx in range(k):
        gen_images[idx].save(os.path.join(args.save_dir, f"gen_{idx:06d}.png"))
    if args.save_grid and k > 0:
        grid_k = min(16, k)
        save_image_grid(gen_images[:grid_k], nrow=4, out_path=os.path.join(args.save_dir, "preview_grid.png"))
        print(f"[Save] Preview grid saved to {os.path.join(args.save_dir, 'preview_grid.png')}")
    print(f"[Save] Saved {k} generated images to {args.save_dir}")

    # -------- Metrics --------
    inc = InceptionWrap(device=device)
    clip_scorer = CLIPScorer(device=device, hf_model_id=args.clip_backbone)

    # Real (reference) moments: full val5k (recommended) or subset
    if args.fid_ref == "full":
        ref_paths = all_val_paths
        cache_path = args.cache_real_stats
    else:
        ref_paths = gen_img_paths
        root, ext = os.path.splitext(args.cache_real_stats)
        cache_path = f"{root}_subset{len(ref_paths)}{ext or '.npz'}"

    real_mu, real_sigma = None, None
    if (not args.recompute_real) and cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        real_mu, real_sigma = d["mu"], d["sigma"]
        print(f"[FID] Loaded cached real moments from {cache_path}.")
    else:
        print("[FID] Computing real MSCOCO moments (this may take a bit, cached afterwards)...")
        real_feats, _ = inc.activations_and_logits_from_paths(ref_paths, batch=64)
        real_mu, real_sigma = compute_moments(real_feats)
        if cache_path:
            np.savez(cache_path, mu=real_mu, sigma=real_sigma)
            print(f"[FID] Saved real moments to {cache_path}.")

    # Optional sanity check — should be ~0
    if args.sanity:
        gen_feats_sanity, _ = inc.activations_and_logits(gen_images, batch=32)
        mu_s, sig_s = compute_moments(gen_feats_sanity)
        fid_self = frechet_distance(mu_s, sig_s, mu_s, sig_s)
        print(f"[Sanity] FID(gen vs gen) = {fid_self:.6f} (expect ~0)")

    print("[Metrics] Inception on generated images ...")
    gen_feats, gen_logits = inc.activations_and_logits(gen_images, batch=32)

    mu_g, sigma_g = compute_moments(gen_feats)
    fid = frechet_distance(real_mu, real_sigma, mu_g, sigma_g)

    splits = min(max(1, args.is_splits), len(gen_images))
    iscore = inception_score_from_logits(gen_logits, splits=splits)
    clip_score = clip_scorer.score(gen_images, [str(t) for t in prompts], batch=32)

    print("\n========== RESULTS (MSCOCO local) ==========")
    print(f"Count                : {len(gen_images)} images")
    print(f"Steps                : {args.steps}")
    print(f"Precision (dtype)    : {args.precision}")
    print(f"Mode                 : {mode_desc}")
    print(f"FID reference        : {'val2017 full' if args.fid_ref=='full' else 'subset'}")
    print(f"FID                  : {fid:.3f}")
    print(f"Inception Score (IS) : {iscore:.3f}")
    print(f"CLIP-Score (CS)      : {clip_score:.3f}")
    print("============================================\n")

if __name__ == "__main__":
    main()



