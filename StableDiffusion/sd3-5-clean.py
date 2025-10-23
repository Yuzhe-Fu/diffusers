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
import logging
from Diffusion_config import (
    setup_model_and_pipeline, 
    setup_scheduler_callback, 
    set_seed,
    ENABLE_MASK_GENERATION,
    GLOBAL_QUANTIZATION_STATE
)

# 如果需要修改全局变量，直接修改导入的模块中的变量
# 这样所有使用这些变量的地方都会保持一致
import Diffusion_config
Diffusion_config.ENABLE_MASK_GENERATION = False  # 修改mask生成控制
# Diffusion_config.GLOBAL_QUANTIZATION_STATE['current_step'] = 0  # 如果需要重置状态


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    # Set random seed
    set_seed(42)
    
    # Setup model and pipeline
    logger.info("Setting up model and pipeline...")
    pipeline = setup_model_and_pipeline()
    
    # Setup scheduler callback
    pipeline = setup_scheduler_callback(pipeline)
    
    SAVE_INFERENCE_DATA = False
    pipeline.transformer.save_inference_data = SAVE_INFERENCE_DATA

    PRINT_INFERENCE_DATA = False  # 设置为True时打印，False时不打印
    pipeline.transformer.print_inference_data = PRINT_INFERENCE_DATA

    # Generate image
    prompt = "a tiny astronaut hatching from an egg on the moon"
    logger.info(f"Generating image with prompt: '{prompt}'")
    
    with torch.no_grad():
        image = pipeline(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    
    # Save the image
    image.save("sd3-5-clean.png")
    logger.info("Image saved as sd3-5-clean.png")

if __name__ == "__main__":
    main()
