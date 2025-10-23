# set the cuda device
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, CLIPTextModelWithProjection
import random
import numpy as np
import pdb
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Set debug level for detailed output
logger.setLevel(logging.INFO)  # Change to logging.DEBUG for debug output

# Global variables to store hidden states and mask
hidden_states_list = []
mask_matrix = None
current_timestep = 0
first_timestep_completed = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def hook_fn(module, input, output):
    """Hook function to capture hidden states from transformer blocks"""
    global hidden_states_list, current_timestep, first_timestep_completed, mask_matrix
    
    # Only capture during the first timestep (timestep 0)
    if current_timestep == 0 and not first_timestep_completed:
        # The output is a tuple (encoder_hidden_states, hidden_states)
        # We want the hidden_states (second element)
        if isinstance(output, tuple) and len(output) == 2:
            hidden_states = output[1]  # Get the hidden_states
            hidden_states_list.append(hidden_states.clone().detach())
            logger.debug(f"Captured hidden states shape: {hidden_states.shape}")
            
            # Check if we have captured all layers (24 layers for SD3.5)
            if len(hidden_states_list) == 24:
                logger.debug("All 24 layers captured, generating mask...")
                mask_matrix = create_mask_from_differences()
                first_timestep_completed = True
                logger.debug(f"Mask generated and ready for use in subsequent timesteps!")
    else:
        pdb.set_trace()

def create_mask_from_differences():
    """
    Create a mask matrix from hidden states differences.
    For each difference between consecutive layers, select the top 20% most changed token positions (per row).
    Returns:
        mask_matrix: torch.Tensor of shape [num_diffs, seq_len], where 1 indicates top 20% change for that row.
    """
    global hidden_states_list, mask_matrix

    if len(hidden_states_list) < 2:
        logger.debug("Not enough hidden states to calculate differences")
        return None

    logger.debug(f"Number of captured hidden states: {len(hidden_states_list)}")

    # Calculate differences between consecutive layers
    differences = []
    for i in range(len(hidden_states_list) - 1):
        diff = (hidden_states_list[i + 1] - hidden_states_list[i]).abs()
        differences.append(diff)
        logger.debug(f"Difference {i} shape: {diff.shape}")

    # For each difference, extract batch=0 and average over hidden_dim to get [seq_len]
    mean_differences = []
    for i, diff in enumerate(differences):
        # diff: [batch, seq_len, hidden_dim]
        batch_0 = diff[0]  # [seq_len, hidden_dim]
        mean_diff = batch_0.mean(dim=-1)  # [seq_len]
        mean_differences.append(mean_diff)
        logger.debug(f"Mean difference {i} shape: {mean_diff.shape}")

    # Stack to get [num_diffs, seq_len]
    mean_differences_tensor = torch.stack(mean_differences)  # [num_diffs, seq_len]
    num_diffs, seq_len = mean_differences_tensor.shape

    # For each row, find top 20% values and set mask
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
        logger.debug(f"Row {i}: {mask_matrix[i].sum().item()} tokens selected as top 20%")

    logger.debug(f"Mask matrix size: {mask_matrix.shape}")
    logger.debug(f"Total number of 1s in mask: {mask_matrix.sum().item()}")
    logger.debug(f"Percentage of 1s: {mask_matrix.sum().item() / mask_matrix.numel() * 100:.2f}%")

    return mask_matrix

quant_config = BitsAndBytesConfig(load_in_8bit=True)
# text_encoder_8bit = T5EncoderModel.from_pretrained(
#     "stabilityai/stable-diffusion-3.5-large",
#     subfolder="text_encoder_3",
#     quantization_config=quant_config,
#     torch_dtype=torch.float16,
# )

text_encoder_8bit = CLIPTextModelWithProjection.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

# Register hooks on all transformer blocks
hooks = []
for i, block in enumerate(transformer_8bit.transformer_blocks):
    hook = block.register_forward_hook(hook_fn)
    hooks.append(hook)
    logger.debug(f"Registered hook on transformer block {i}")

pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    # device_map="balanced",
)
# pipeline = pipeline.to("cuda")
pipeline.enable_model_cpu_offload()

# Custom scheduler callback to track timesteps
# We need to track when we're in the first timestep vs subsequent timesteps
original_step = pipeline.scheduler.step

def custom_step(*args, **kwargs):
    global current_timestep, first_timestep_completed
    
    # Get the current timestep before the step
    if len(pipeline.scheduler.timesteps) > 0:
        current_timestep = pipeline.scheduler.timesteps[0].item()
        logger.debug(f"Current timestep: {current_timestep}")
    
    result = original_step(*args, **kwargs)
    
    # After the step, check if we've moved to the next timestep
    if len(pipeline.scheduler.timesteps) > 0:
        new_timestep = pipeline.scheduler.timesteps[0].item()
        if new_timestep != current_timestep:
            logger.debug(f"Timestep changed from {current_timestep} to {new_timestep}")
            current_timestep = new_timestep
    
    return result

pipeline.scheduler.step = custom_step

prompt = "a tiny astronaut hatching from an egg on the moon"

# Generate image with custom callback
# The mask will be generated automatically during the first timestep
with torch.no_grad():
    image = pipeline(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]

# Store mask for later use (mask was already generated during first timestep)
if mask_matrix is not None:
    torch.save(mask_matrix, "attention_mask.pt")
    logger.info("Mask matrix saved to attention_mask.pt")
else:
    logger.warning("No mask matrix was generated!")

# Clean up hooks
for hook in hooks:
    hook.remove()

# Save the image
image.save("sd3-5.png")