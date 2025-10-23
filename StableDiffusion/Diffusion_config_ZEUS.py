# SD3.5 Global Variable Based Selective Quantization Implementation
# This code implements the following workflow:
# 1. In step 0: Capture input hidden_states of each JointTransformerBlock via global variables
# 2. Calculate difference values between adjacent layers and generate mask matrix
# 3. In subsequent steps: Use mask matrix to selectively quantize JointTransformerBlock inputs

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, CLIPTextModelWithProjection
import random
import numpy as np
import logging
import pdb
# Set up logging
# You can configure the log file output path by setting the 'filename' parameter in logging.basicConfig.
# For example, to change the log file location, modify the 'filename' argument below:
import os

LOG_FILE_PATH = "./StableDiffusion/logs/log.txt"  # Change this path to your desired log file location

# Ensure the log directory exists before configuring logging
log_dir = os.path.dirname(LOG_FILE_PATH)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=LOG_FILE_PATH
)
logger = logging.getLogger(__name__)

# Global control variable for mask generation
ENABLE_MASK_GENERATION = False  # Set to False to skip mask generation during first timestep

# Global variables for quantization state management
GLOBAL_QUANTIZATION_STATE = {
    'current_step': 0,
    'first_step_completed': False,
    'layer_inputs': [],  # Store input hidden states for each layer
    'mask_matrix': None,
    'layer_counter': 0
}

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_mask_from_adjacent_differences():
    """
    Create a mask matrix from adjacent layer hidden states differences.
    For each pair of adjacent layers, calculate layer[i+1] - layer[i] to get the transformation.
    Select the top 20% most changed token positions for each layer.
    Returns:
        mask_matrix: torch.Tensor of shape [num_layers-1, seq_len], where 1 indicates top 20% change for that layer.
    """
    global GLOBAL_QUANTIZATION_STATE
    
    layer_inputs = GLOBAL_QUANTIZATION_STATE['layer_inputs']
    
    logger.info(f"Creating mask from adjacent differences. Inputs: {len(layer_inputs)}")
    
    if len(layer_inputs) < 2:
        logger.error(f"Not enough data to calculate adjacent differences. Inputs: {len(layer_inputs)}")
        return None

    num_layers = len(layer_inputs)
    logger.info(f"Number of layers: {num_layers}")

    # Calculate differences between adjacent layers: layer[i+1] - layer[i]
    differences = []
    for i in range(num_layers - 1):
        diff = (layer_inputs[i + 1] - layer_inputs[i]).abs()
        differences.append(diff)
        logger.debug(f"Adjacent difference {i} (layer {i+1} - layer {i}) shape: {diff.shape}")

    # For each difference, extract batch=0 and average over hidden_dim to get [seq_len]
    mean_differences = []
    for i, diff in enumerate(differences):
        # diff: [batch, seq_len, hidden_dim]
        batch_0 = diff[0]  # [seq_len, hidden_dim]
        mean_diff = batch_0.mean(dim=-1)  # [seq_len]
        mean_differences.append(mean_diff)
        logger.debug(f"Mean adjacent difference {i} shape: {mean_diff.shape}")

    # Stack to get [num_layers-1, seq_len]
    mean_differences_tensor = torch.stack(mean_differences)  # [num_layers-1, seq_len]
    num_diffs, seq_len = mean_differences_tensor.shape

    # For each row (adjacent layer difference), find top 20% values and set mask
    mask_matrix = torch.zeros_like(mean_differences_tensor)
    top_k = max(1, int(0.2 * seq_len))  # At least 1

    for i in range(num_diffs):
        row = mean_differences_tensor[i]
        if top_k >= len(row):
            mask_matrix[i] = 1.0
        else:
            # Get indices of top 20% values
            topk_indices = torch.topk(row, top_k).indices
            mask_matrix[i, topk_indices] = 1.0
        logger.info(f"Adjacent difference {i}: {mask_matrix[i].sum().item()} tokens selected as top 20%")

    logger.info(f"Mask matrix size: {mask_matrix.shape}")
    logger.info(f"Total number of 1s in mask: {mask_matrix.sum().item()}")
    logger.info(f"Percentage of 1s: {mask_matrix.sum().item() / mask_matrix.numel() * 100:.2f}%")

    return mask_matrix


def setup_model_and_pipeline():
    """Setup quantized model and pipeline"""
    # Setup text encoder with quantization
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder_8bit = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        subfolder="text_encoder",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    # Setup transformer with quantization
    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    # Create pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        text_encoder=text_encoder_8bit,
        transformer=transformer_8bit,
        torch_dtype=torch.float16,
    )
    pipeline.enable_model_cpu_offload()
    
    return pipeline

def setup_scheduler_callback(pipeline):
    """Setup custom scheduler callback to track timesteps"""
    original_step = pipeline.scheduler.step

    def custom_step(*args, **kwargs):
        global GLOBAL_QUANTIZATION_STATE
        
        # Log current step
        logger.debug(f"Starting inference step {GLOBAL_QUANTIZATION_STATE['current_step']}")
        
        result = original_step(*args, **kwargs)
        
        # After the step, increment step counter and reset layer index
        GLOBAL_QUANTIZATION_STATE['current_step'] += 1
        GLOBAL_QUANTIZATION_STATE['layer_counter'] = 0
        
        logger.debug(f"Completed inference step {GLOBAL_QUANTIZATION_STATE['current_step']-1}, moving to step {GLOBAL_QUANTIZATION_STATE['current_step']}")
        return result

    pipeline.scheduler.step = custom_step
    return pipeline

# Global control variable for timestep skipping
ENABLE_TIMESTEP_SKIPPING = False  # Set to True to enable timestep skipping
ENABLE_MASK = False  # Set to True to enable mask on outliers in input noise_pred

# Global variables for timestep skipping state management
GLOBAL_TIMESTEP_SKIP_STATE = {
    'noise_pred_cache': [],  # Store noise_pred from previous timesteps
    'cache_step': [],        # Store step index from previous timesteps
    'current_step': 0,
    'initial_normal_steps': 2,  # Number of initial steps to keep normal computation
    'skip_end_step': 44,        # Stop skipping at step 44 (0-indexed)
    'total_steps': 50,          # Total number of inference steps
    'skipped_timesteps': []     # Store indices of skipped timesteps
}

def should_skip_timestep(step_index):
    """
    Determine if current timestep should be skipped based on the pattern:
    - First N steps: normal computation
    - After that: every 3 steps form a cycle
    - (i-N)%3==0: Skip, use reuse
    - (i-N)%3==1: Normal computation  
    - (i-N)%3==2: Skip, use linear interpolation
    """
    global GLOBAL_TIMESTEP_SKIP_STATE
    
    if not ENABLE_TIMESTEP_SKIPPING:
        return False
    
    initial_steps = GLOBAL_TIMESTEP_SKIP_STATE['initial_normal_steps']
    skip_end = GLOBAL_TIMESTEP_SKIP_STATE['skip_end_step']
    
    # Don't skip if before initial steps or after skip end
    if step_index < initial_steps or step_index > skip_end:
        return False
    
    # Use simplified logic: (i-N)%3
    relative_step = step_index - initial_steps
    cycle_position = relative_step % 3
    
    # Skip positions 0 and 2, keep position 1
    return cycle_position == 0 or cycle_position == 1

def get_skipped_noise_pred(step_index):
    """
    Get noise_pred for skipped timestep using interpolation or reuse.
    Pattern:
    - (i-N)%3==0: Skip, use reuse (t_current = t_prev)
    - (i-N)%3==2: Skip, use linear interpolation (t_current = 2*t_prev - t_prev_prev)
    """
    global GLOBAL_TIMESTEP_SKIP_STATE
    # pdb.set_trace()
    cache = GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache']
    cache_step = GLOBAL_TIMESTEP_SKIP_STATE['cache_step']
    initial_steps = GLOBAL_TIMESTEP_SKIP_STATE['initial_normal_steps']
    
    if len(cache) < 2:
        raise ValueError("Not enough cached noise_pred values for interpolation")
    
    # Use simplified logic: (i-N)%3
    relative_step = step_index - initial_steps
    cycle_position = relative_step % 3
    print(f"step_index: {step_index}")
    if cycle_position == 1:  # Skip, use reuse
        GLOBAL_TIMESTEP_SKIP_STATE['skipped_timesteps'].append(step_index)
        print(f"step_index: {step_index}, use reuse with  {cache_step[-2]}")
        return cache[-2]
        # print(f"step_index: {step_index}, use psi with  2*{cache_step[-1]} - {cache_step[-2]}")
        # return 2 * cache[-1] - cache[-2]
    elif cycle_position == 0:  # Skip, use linear interpolation
        GLOBAL_TIMESTEP_SKIP_STATE['skipped_timesteps'].append(step_index)
        # print(f"step_index: {step_index}, use psi with  2*{cache_step[-1]} - {cache_step[-2]}")
        # return 2 * cache[-1] - cache[-2]
        print(f"step_index: {step_index}, use reuse with  {cache_step[-2]}")
        return cache[-2]
    else:
        raise ValueError(f"Invalid skip pattern for step {step_index}, cycle position {cycle_position}")

def cache_noise_pred(noise_pred, step_index):
    """Cache noise_pred for future interpolation/reuse"""
    global GLOBAL_TIMESTEP_SKIP_STATE
    
    # Keep only the last 2 values to save memory
    cache = GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache']
    cache_step = GLOBAL_TIMESTEP_SKIP_STATE['cache_step']
    cache.append(noise_pred.clone().detach())
    cache_step.append(step_index)
    
    if len(cache) > 2:
        cache.pop(0)  # Remove oldest value
        cache_step.pop(0)
    
    GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache'] = cache
    GLOBAL_TIMESTEP_SKIP_STATE['cache_step'] = cache_step

def record_skipped_timestep(step_index):
    """Record the index of a skipped timestep"""
    global GLOBAL_TIMESTEP_SKIP_STATE
    
def get_mask(step_index):
    """Get mask for outliers in noise_pred"""
    global GLOBAL_TIMESTEP_SKIP_STATE
    cache = GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache']
    
    if len(cache) == 0:
        print("Warning: cache is empty, returning zero mask")
        return torch.zeros(1, 16, 128, 128)
    
    # 获取最新的 noise_pred
    noise_pred = cache[-1]  # [1, 16, 128, 128]
    
    # 展平张量以便进行 topk 操作
    flat_noise = noise_pred.abs().flatten()  # [1*16*128*128]
    
    # 计算要选择的 top-k 数量
    total_elements = flat_noise.numel()
    top_k = max(1, int(0.05 * total_elements))
    
    top_k = min(top_k, total_elements)
    _, topk_indices = torch.topk(flat_noise, top_k)
    flat_mask = torch.zeros_like(flat_noise)  # [1*16*128*128]
    flat_mask[topk_indices] = 1.0
    
    # 将 flat_mask 重新整形为与 cache[-1] 相同的尺寸
    mask = flat_mask.reshape(noise_pred.shape)  # [1, 16, 128, 128]
    
    # print(f"Generated mask for outliers: {top_k}/{total_elements} elements selected")
    # print(f"Mask shape: {mask.shape}, cache[-1] shape: {noise_pred.shape}")
    return mask



def get_differential_mask(step_index):
    """Get mask for outliers in noise_pred"""
    global GLOBAL_TIMESTEP_SKIP_STATE
    cache = GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache']
    
    # 获取最新的 noise_pred
    noise_pred = cache[-1]-cache[-2]  # [1, 16, 128, 128]
    
    # 展平张量以便进行 topk 操作
    flat_noise = noise_pred.abs().flatten()  # [1*16*128*128]
    
    # 计算要选择的 top-k 数量
    total_elements = flat_noise.numel()
    top_k = max(1, int(0.05 * total_elements))
    
    top_k = min(top_k, total_elements)
    _, topk_indices = torch.topk(flat_noise, top_k)
    flat_mask = torch.zeros_like(flat_noise)  # [1*16*128*128]
    flat_mask[topk_indices] = 1.0
    
    # 将 flat_mask 重新整形为与 cache[-1] 相同的尺寸
    mask = flat_mask.reshape(noise_pred.shape)  # [1, 16, 128, 128]
    
    # print(f"Generated mask for outliers: {top_k}/{total_elements} elements selected")
    # print(f"Mask shape: {mask.shape}, cache[-1] shape: {noise_pred.shape}")
    return mask