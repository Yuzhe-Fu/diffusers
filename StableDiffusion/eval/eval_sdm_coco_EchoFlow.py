# eval_sdm_coco_EchoFlow.py
# SD v3.5 (DiT) with EchoFlow optimization on local MSCOCO val2017.
# Computes FID/IS/CLIP-Score, saves a subset of generated images and an optional grid.

'''
# EchoFlow baseline (FP16 everywhere)
CUDA_VISIBLE_DEVICES=3 python eval_sdm_coco_EchoFlow.py --mscoco_root /home/yf184/mscoco --model stabilityai/stable-diffusion-3.5-medium --num_samples 100 --steps 50 --batch 8 --precision fp16 --height 1024 --width 1024 --fid_ref full --save_dir sdm35_mscoco_baseline --save_k 20 --save_grid --baseline

# EchoFlow with timestep skipping
CUDA_VISIBLE_DEVICES=3 python eval_sdm_coco_EchoFlow.py --mscoco_root /home/yf184/mscoco --model stabilityai/stable-diffusion-3.5-medium --num_samples 100 --steps 50 --batch 8 --precision fp16 --height 1024 --width 1024 --fid_ref full --save_dir sdm35_mscoco_echoflow --save_k 20 --save_grid --enable_timestep_skipping --enable_mask --topk_ratio 0.1 --skip_steps 7 49 3 1 --hidden_mask
'''

import argparse, os, sys, math
from typing import List
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image

# Add local diffusers directory to Python path
local_diffusers_path = '/home/yf184/diffusers'
if local_diffusers_path not in sys.path:
    sys.path.insert(0, local_diffusers_path)

from mscoco_local import MSCOCOVal2017
from metrics_eval import (
    InceptionWrap, compute_moments, frechet_distance,
    inception_score_from_logits, CLIPScorer
)

# Import EchoFlow modules
local_diffusers_path = '/home/yf184/diffusers/StableDiffusion'
if local_diffusers_path not in sys.path:
    sys.path.insert(0, local_diffusers_path)

from Diffusion_config import (
    setup_model_and_pipeline, 
    setup_model_and_pipeline_no_quant,
    set_seed,
    ENABLE_MASK_GENERATION,
    ENABLE_TIMESTEP_SKIPPING,
    GLOBAL_TIMESTEP_SKIP_STATE
)
import Diffusion_config

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
    ap.add_argument("--baseline", action="store_true", help="Run pure baseline without EchoFlow optimization")
    
    # EchoFlow specific parameters
    ap.add_argument("--enable_timestep_skipping", action="store_true", 
                    help="Enable EchoFlow timestep skipping optimization")
    ap.add_argument("--enable_mask", action="store_true", 
                    help="Enable EchoFlow mask-based selective quantization")
    ap.add_argument("--topk_ratio", type=list, default=[0.01]*50,
                    help="Top-k ratio for EchoFlow selective quantization")
    ap.add_argument("--skip_steps", type=int, nargs=4, default=[7, 49, 3, 1],
                    help="EchoFlow timestep skipping configuration [start, end, loop_length, layer_interval]")
    ap.add_argument("--hidden_mask", action="store_true", default=True,
                    help="Enable EchoFlow hidden dimension masking")
    ap.add_argument("--context_mask", action="store_true", default=False,
                    help="Enable EchoFlow context masking")
    ap.add_argument("--mask_generatedbyinput", action="store_true", default=False,
                    help="Enable EchoFlow mask generated by input, with accumulated mask logic")
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

    # -------- Build SD pipeline --------
    if args.baseline:
        # Standard baseline mode
        pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype).to(device)
        if hasattr(pipe, "safety_checker"):
            try:
                pipe.safety_checker = None
            except Exception:
                pass
        mode_desc = "baseline"
        print(f"[Mode] Baseline (no EchoFlow optimization)")
        print(f"[Pipeline] Loaded {args.model}; steps = {args.steps} | dtype = {dtype}")
        
    elif args.enable_timestep_skipping:
        # EchoFlow mode - use custom pipeline
        print("[Pipeline] Setting up EchoFlow pipeline...")
        
        # Reset global state
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['TS_STATE'] = []
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache'] = []
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['cache_step'] = []
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['current_step'] = 0
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['skipped_timesteps'] = []
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['all_noise_pred_cache'] = []
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['all_cache_step'] = []
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'] = []
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['context_mask_cache'] = []
        Diffusion_config.AllTSDataList = []
        Diffusion_config.GLOBAL_GETMASK_LAYER = []
        
        # Configure EchoFlow parameters
        Diffusion_config.ENABLE_TIMESTEP_SKIPPING = True
        Diffusion_config.ENABLE_MASK = args.enable_mask
        Diffusion_config.Hidden_MASK = args.hidden_mask
        Diffusion_config.Context_MASK = args.context_mask
        Diffusion_config.TOPK_RATIO = args.topk_ratio
        Diffusion_config.MaskGeneratedbyInput = args.mask_generatedbyinput
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['skip_steps'] = args.skip_steps
        Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['total_steps'] = args.steps
        
        # Generate timestep skipping state
        Diffusion_config.generate_ts_state()
        Diffusion_config.generate_GLOBAL_GETMASK_LAYER()
        
        # Use EchoFlow pipeline
        # pipe = setup_model_and_pipeline()
        pipe = setup_model_and_pipeline_no_quant()
        mode_desc = f"EchoFlow (topk_ratio={args.topk_ratio}, skip_steps={args.skip_steps})"
        print(f"[Mode] {mode_desc}")
        
    else:
        # Default to baseline if no specific mode is selected
        pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype).to(device)
        if hasattr(pipe, "safety_checker"):
            try:
                pipe.safety_checker = None
            except Exception:
                pass
        mode_desc = "baseline"
        print(f"[Mode] Baseline (no EchoFlow optimization)")
        print(f"[Pipeline] Loaded {args.model}; steps = {args.steps} | dtype = {dtype}")

    # -------- Generate images --------
    os.makedirs(args.save_dir, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    gen_images: List[Image.Image] = []
    print("[Gen] Generating images with Stable Diffusion...")
    
    for i in range(0, len(prompts), args.batch):
        batch_prompts = prompts[i: i + args.batch]
        
        # Reset seed for each batch to ensure reproducibility
        set_seed(args.seed + i)
        
        with torch.inference_mode():
            if args.enable_timestep_skipping:
                # EchoFlow pipeline
                out = pipe(
                    batch_prompts,
                    num_inference_steps=Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['total_steps'],
                    guidance_scale=args.guidance,
                    output_type='pt'
                )
                # Convert tensor to PIL images
                imgs = []
                for img_tensor in out.images:
                    # Convert tensor to PIL Image
                    if isinstance(img_tensor, torch.Tensor):
                        # Normalize from [-1, 1] to [0, 1]
                        img_tensor = (img_tensor + 1) / 2
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        # Convert to PIL
                        img_pil = Image.fromarray((img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                        imgs.append(img_pil.convert("RGB"))
                    else:
                        imgs.append(img_tensor.convert("RGB"))
            else:
                # Standard pipeline
                out = pipe(
                    batch_prompts,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                    width=args.width, height=args.height,
                )
                imgs = [im.convert("RGB") for im in out.images]
        
        gen_images.extend(imgs)
        
        # Clean up memory for EchoFlow
        if args.enable_timestep_skipping:
            torch.cuda.empty_cache()

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
    
    # Skip CLIP scoring if there are PyTorch version issues
    try:
        clip_scorer = CLIPScorer(device=device, hf_model_id=args.clip_backbone)
        clip_available = True
    except Exception as e:
        print(f"[Warning] CLIP scoring unavailable due to: {e}")
        print("[Warning] Skipping CLIP score calculation")
        clip_scorer = None
        clip_available = False

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
    
    if clip_available:
        clip_score = clip_scorer.score(gen_images, [str(t) for t in prompts], batch=32)
    else:
        clip_score = 0.0

    print("\n========== RESULTS (MSCOCO local) ==========")
    print(f"Count                : {len(gen_images)} images")
    print(f"Steps                : {args.steps}")
    print(f"Precision (dtype)    : {args.precision}")
    print(f"Mode                 : {mode_desc}")
    print(f"FID reference        : {'val2017 full' if args.fid_ref=='full' else 'subset'}")
    print(f"FID                  : {fid:.3f}")
    print(f"Inception Score (IS) : {iscore:.3f}")
    if clip_available:
        print(f"CLIP-Score (CS)      : {clip_score:.3f}")
    else:
        print(f"CLIP-Score (CS)      : N/A (PyTorch version issue)")
    print("============================================\n")

if __name__ == "__main__":
    main()



