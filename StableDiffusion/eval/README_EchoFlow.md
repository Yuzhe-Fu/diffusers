# EchoFlow Evaluation Script

This script (`eval_sdm_coco_EchoFlow.py`) evaluates the EchoFlow optimization method on MSCOCO dataset for FID/IS/CLIP-Score metrics.

## Features

- **Baseline Mode**: Standard Stable Diffusion 3.5 without optimization
- **EchoFlow Mode**: With timestep skipping and selective quantization
- **FID/IS/CLIP-Score**: Comprehensive evaluation metrics
- **Image Saving**: Saves generated images and preview grids

## Usage

### 1. Prepare MSCOCO Dataset

First, prepare the MSCOCO dataset:

```bash
python mscoco_local.py --root /home/yf184/mscoco
```

### 2. Run Baseline Evaluation

```bash
CUDA_VISIBLE_DEVICES=2 python eval_sdm_coco_EchoFlow.py \
    --mscoco_root /home/yf184/diffusers/StableDiffusion/eval/dataset/mscoco \
    --model stabilityai/stable-diffusion-3.5-medium \
    --num_samples 100 \
    --steps 50 \
    --batch 8 \
    --precision fp16 \
    --height 1024 \
    --width 1024 \
    --fid_ref full \
    --save_dir sdm35_mscoco_baseline \
    --save_k 20 \
    --save_grid \
    --baseline
```

### 3. Run EchoFlow Evaluation

```bash
CUDA_VISIBLE_DEVICES=2 python eval_sdm_coco_EchoFlow.py \
    --mscoco_root /home/yf184/diffusers/StableDiffusion/eval/dataset/mscoco \
    --model stabilityai/stable-diffusion-3.5-medium \
    --num_samples 100 \
    --steps 50 \
    --batch 1 \
    --precision fp16 \
    --height 1024 \
    --width 1024 \
    --fid_ref full \
    --save_dir sdm35_mscoco_echoflow-topk0 \
    --save_k 20 \
    --save_grid \
    --enable_timestep_skipping \
    --enable_mask \
    --skip_steps 7 49 3 1 \
    --hidden_mask
```
    # --topk_ratio [0]*50 \
## Parameters

### EchoFlow Specific Parameters

- `--enable_timestep_skipping`: Enable EchoFlow timestep skipping optimization
- `--enable_mask`: Enable EchoFlow mask-based selective quantization
- `--topk_ratio FLOAT`: Top-k ratio for selective quantization (default: 0.1)
- `--skip_steps INT INT INT INT`: Timestep skipping configuration [start, end, loop_length, layer_interval] (default: 7 49 3 1)
- `--hidden_mask`: Enable hidden dimension masking (default: True)
- `--context_mask`: Enable context masking (default: False)

### Standard Parameters

- `--mscoco_root PATH`: Path to MSCOCO dataset root
- `--num_samples INT`: Number of images to evaluate (default: 5000)
- `--steps INT`: Number of denoising steps (default: 50)
- `--batch INT`: Batch size (default: 4)
- `--precision STR`: Precision mode (fp32/fp16/bf16, default: fp32)
- `--save_dir PATH`: Directory to save generated images
- `--save_k INT`: Number of images to save (default: 64)
- `--save_grid`: Save preview grid

## Output

The script will output:

1. **Generated Images**: Saved in the specified `--save_dir`
2. **Preview Grid**: `preview_grid.png` (if `--save_grid` is enabled)
3. **Evaluation Metrics**:
   - FID (Fréchet Inception Distance)
   - IS (Inception Score)
   - CLIP-Score (if available)

## Example Output

```
========== RESULTS (MSCOCO local) ==========
Count                : 100 images
Steps                : 50
Precision (dtype)    : fp16
Mode                 : EchoFlow (topk_ratio=0.1, skip_steps=[7, 49, 3, 1])
FID reference        : val2017 full
FID                  : 12.345
Inception Score (IS) : 8.123
CLIP-Score (CS)      : 0.789
============================================
```

## Notes

- The script automatically handles PyTorch version compatibility issues with CLIP scoring
- Memory is automatically cleaned up after each batch for EchoFlow mode
- Global state is reset before each run to ensure reproducibility
- The script supports both tensor and PIL image outputs from the EchoFlow pipeline
