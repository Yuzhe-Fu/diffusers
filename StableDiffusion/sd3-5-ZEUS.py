# SD3.5 Global Variable Based Selective Quantization Implementation
# This code implements the following workflow:
# 1. In step 0: Capture input hidden_states of each JointTransformerBlock via global variables
# 2. Calculate difference values between adjacent layers and generate mask matrix
# 3. In subsequent steps: Use mask matrix to selectively quantize JointTransformerBlock inputs

import sys
import os

# Add local diffusers directory to Python path
local_diffusers_path = '/home/yf184/diffusers'
if local_diffusers_path not in sys.path:
    sys.path.insert(0, local_diffusers_path)

import torch
import lpips

from torchvision.utils import save_image
import torchvision.transforms as T

import logging
from Diffusion_config import (
    setup_model_and_pipeline, 
    setup_scheduler_callback, 
    set_seed,
    ENABLE_MASK_GENERATION,
    GLOBAL_QUANTIZATION_STATE,
    ENABLE_TIMESTEP_SKIPPING,
    GLOBAL_TIMESTEP_SKIP_STATE
)

# 如果需要修改全局变量，直接修改导入的模块中的变量
# 这样所有使用这些变量的地方都会保持一致
import Diffusion_config

Diffusion_config.ENABLE_MASK_GENERATION = False  # 修改mask生成控制
Diffusion_config.ENABLE_TIMESTEP_SKIPPING = False  # 启用timestep跳除功能
SAVE_INFERENCE_DATA = False # 设置为True时保存inference数据，False时不保存
PRINT_INFERENCE_DATA = False  # 设置为True时打印timestep信息，False时不打印

# Set up logging
# Reuse the logger from Diffusion_config for consistent logging
logger = Diffusion_config.logger


def main():
    """Main execution function"""
    # Set random seed
    set_seed(42)
    Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['initial_normal_steps'] = 5
    Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['skip_end_step'] = 44
    Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['total_steps'] = 50
    Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache'] = []
    Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['current_step'] = 0
    Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['skipped_timesteps'] = []

    Diffusion_config.ENABLE_TIMESTEP_SKIPPING = False  # 启用timestep跳除功能
    Diffusion_config.ENABLE_MASK = False
    pipeline = setup_model_and_pipeline()

    # pipeline = setup_scheduler_callback(pipeline)

    # prompt = "a tiny astronaut hatching from an egg on the moon"
    prompt = "Cinematic photograph of a dark cat walking in the fantasy moonlight garden"
    logger.info(f"Generating image with prompt: '{prompt}'")
    
    with torch.no_grad():
        ori_output = pipeline(prompt, num_inference_steps=Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['total_steps'], guidance_scale=7.0, output_type='pt').images
    del pipeline
    torch.cuda.empty_cache()


    Diffusion_config.ENABLE_TIMESTEP_SKIPPING = True
    Diffusion_config.ENABLE_MASK = False
    
    # Log timestep skipping configuration
    if Diffusion_config.ENABLE_TIMESTEP_SKIPPING:
        logger.info("Timestep skipping is ENABLED")
        initial_steps = Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['initial_normal_steps']
        skip_end = Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['skip_end_step']
        logger.info(f"Initial normal steps: {initial_steps}")
        logger.info(f"Skip range: steps {initial_steps} to {skip_end}")
        logger.info(f"Enable mask: {Diffusion_config.ENABLE_MASK}")
    else:
        logger.info("Timestep skipping is DISABLED")
    
    # Setup model and pipeline
    logger.info("Setting up model and pipeline...")
    pipeline = setup_model_and_pipeline()
    # Setup scheduler callback
    # pipeline = setup_scheduler_callback(pipeline)

    # Generate image
    # prompt = "a tiny astronaut hatching from an egg on the moon"
    prompt = "Cinematic photograph of a dark cat walking in the fantasy moonlight garden"
    logger.info(f"Generating image with prompt: '{prompt}'")
    
    set_seed(42)
    with torch.no_grad():
        cap_output = pipeline(prompt, num_inference_steps=Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['total_steps'], guidance_scale=7.0, output_type='pt').images
    

    save_image([ori_output[0], cap_output[0]], "./StableDiffusion/logs/output.png")
    logging.info("Saved to output.png. Done!")

    
    # Log final cache state and skipped timesteps
    skipped_timesteps = Diffusion_config.GLOBAL_TIMESTEP_SKIP_STATE['skipped_timesteps']
    logger.info(f"Skipped timesteps: {skipped_timesteps}")
    logger.info(f"Total skipped timesteps: {len(skipped_timesteps)}")

    logging.info("Evaluating LPIPS")
    p_r = torch.stack([T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])(img) for img in ori_output]).to('cuda')

    p_o = torch.stack([T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])(img) for img in cap_output]).to('cuda')

    loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
    d = loss_fn_alex(p_r, p_o)
    logging.info(f"LPIPS: {d.item()}")
    print(f"LPIPS: {d.item()}")

if __name__ == "__main__":
    main()
