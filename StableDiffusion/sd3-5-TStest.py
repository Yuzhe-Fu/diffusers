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
logger.setLevel(logging.INFO)  # Change to INFO/DEBUG

# Global variables to store hidden states and mask
layer_inputs = []  # Store input hidden states for each layer
layer_outputs = []  # Store output hidden states for each layer
mask_matrix = None
current_step = 0
first_step_completed = False

# For subsequent steps, store a flag and a per-layer counter
subsequent_step_processing = False
subsequent_step_layer_idx = 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def quantize_tensor_to_int8_approx(tensor):
    """
    Approximate a float16 tensor to int8 values, but keep dtype as float16.
    This simulates int8 quantization for the values, but keeps the tensor as fp16 for compatibility.
    """
    # Clamp to int8 range
    min_val, max_val = -128, 127
    # Find scale for symmetric quantization
    max_abs = tensor.abs().max()
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / max_val
    # Quantize to int8, then dequantize back to float16
    tensor_int8 = torch.clamp((tensor / scale).round(), min_val, max_val)
    tensor_fp16_approx = (tensor_int8 * scale).to(dtype=tensor.dtype)
    return tensor_fp16_approx

def transformer_pre_hook_fn(module, input):
    """
    Pre-hook function for transformer level to capture hidden states.
    This should capture the main hidden states that flow through the transformer.
    """
    global current_step, first_step_completed, mask_matrix
    global layer_inputs, layer_outputs

    # Debug: Log transformer pre-hook call
    logger.info(f"Transformer pre-hook called: step={current_step}, first_completed={first_step_completed}")
    logger.info(f"Transformer input type: {type(input)}, is tuple: {isinstance(input, tuple)}")
    if isinstance(input, tuple):
        logger.info(f"Transformer input tuple length: {len(input)}")
        for i, inp in enumerate(input):
            logger.info(f"Input {i}: type={type(inp)}, shape={getattr(inp, 'shape', 'no shape')}")
    else:
        logger.info(f"Transformer input is not a tuple: {input}")
    
    # Cache input during first step
    if current_step == 0 and not first_step_completed:
        if isinstance(input, tuple) and len(input) > 0:
            # Try to find hidden_states in the input tuple
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor) and len(inp.shape) == 3:  # [batch, seq_len, hidden_dim]
                    logger.info(f"Found potential hidden states at input[{i}]: shape={inp.shape}")
                    layer_inputs.append(inp.clone().detach())
                    
                    # Check if we have captured all inputs and can generate mask
                    if len(layer_inputs) == 24 and len(layer_outputs) == 24:
                        logger.info("All 24 inputs and outputs captured, generating mask...")
                        mask_matrix = create_mask_from_differences()
                        first_step_completed = True
                        logger.info(f"Mask generated successfully! Shape: {mask_matrix.shape if mask_matrix is not None else 'None'}")
                        logger.info(f"Mask ready for use in subsequent steps!")
                    elif len(layer_inputs) == 24:
                        logger.info(f"Inputs captured (24), but outputs only: {len(layer_outputs)}. Waiting for outputs...")
                    break

def transformer_post_hook_fn(module, input, output):
    """
    Post-hook function for transformer level to capture output hidden states.
    """
    global current_step, first_step_completed, mask_matrix
    global layer_inputs, layer_outputs

    # Debug: Log transformer post-hook call
    logger.info(f"Transformer post-hook called: step={current_step}, first_completed={first_step_completed}")
    logger.info(f"Transformer output type: {type(output)}, is tuple: {isinstance(output, tuple)}")
    if isinstance(output, tuple):
        logger.info(f"Transformer output tuple length: {len(output)}")
        for i, out in enumerate(output):
            logger.info(f"Output {i}: type={type(out)}, shape={getattr(out, 'shape', 'no shape')}")
    else:
        logger.info(f"Transformer output is not a tuple: {output}")
    
    # Cache output during first step
    if current_step == 0 and not first_step_completed:
        if isinstance(output, tuple) and len(output) > 0:
            # Try to find hidden_states in the output tuple
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and len(out.shape) == 3:  # [batch, seq_len, hidden_dim]
                    logger.info(f"Found potential hidden states at output[{i}]: shape={out.shape}")
                    layer_outputs.append(out.clone().detach())
                    
                    # Check if we have captured all outputs and can generate mask
                    if len(layer_inputs) == 24 and len(layer_outputs) == 24:
                        logger.info("All 24 inputs and outputs captured, generating mask...")
                        mask_matrix = create_mask_from_differences()
                        first_step_completed = True
                        logger.info(f"Mask generated successfully! Shape: {mask_matrix.shape if mask_matrix is not None else 'None'}")
                        logger.info(f"Mask ready for use in subsequent steps!")
                    elif len(layer_outputs) == 24:
                        logger.info(f"Outputs captured (24), but inputs only: {len(layer_inputs)}. Waiting for inputs...")
                    break

def block_pre_hook_fn(module, input):
    """
    Pre-hook function for individual transformer blocks.
    This is a simplified version that just logs the call.
    """
    global current_step, first_step_completed, mask_matrix
    global subsequent_step_processing, subsequent_step_layer_idx
    global layer_inputs, layer_outputs

    # Debug: Log every block pre-hook call
    logger.info(f"Block pre-hook called: step={current_step}, first_completed={first_step_completed}, inputs_count={len(layer_inputs)}")
    
    # Apply quantization during steps after the first one
    if current_step > 0 and first_step_completed and mask_matrix is not None:
        logger.info(f"Applying quantization in step {current_step}, layer {subsequent_step_layer_idx}")
        # Get the mask for this layer
        num_layers, seq_len = mask_matrix.shape
        # Now we have 24 masks for 24 layers
        mask_idx = min(subsequent_step_layer_idx, num_layers - 1)
        mask = mask_matrix[mask_idx]  # shape: [seq_len]
        subsequent_step_layer_idx += 1

        # Get hidden_states from input (first element of input tuple)
        if isinstance(input, tuple) and len(input) > 0:
            hidden_states = input[0]  # shape: [batch, seq_len, hidden_dim]
            if hidden_states is not None and isinstance(hidden_states, torch.Tensor):
                # Apply mask-based selective quantization
                # mask: [seq_len], expand to [batch, seq_len, hidden_dim]
                mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand(
                    hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
                )  # [batch, seq_len, hidden_dim]
                mask_fp16 = mask_expanded.to(dtype=hidden_states.dtype)
                
                # Quantize the hidden states
                quantized_part = quantize_tensor_to_int8_approx(hidden_states)
                logger.info(f"Quantized successfully for layer {subsequent_step_layer_idx-1}")
                
                # Apply selective quantization: keep original where mask=1, quantize where mask=0
                quantized_hidden_states = mask_fp16 * hidden_states + (1.0 - mask_fp16) * quantized_part
                
                # Replace the first element of input tuple with quantized version
                new_input = (quantized_hidden_states,) + input[1:]
                return new_input
    else:
        # Debug: Print why quantization is not applied
        logger.info(f"Quantization not applied: step={current_step}, first_completed={first_step_completed}, mask_exists={mask_matrix is not None}")

def post_hook_fn(module, input, output):
    """
    Post-hook function to cache output hidden_states during the first step.
    This runs AFTER the transformer block processes the input.
    """
    global layer_outputs, current_step, first_step_completed, mask_matrix

    # Debug: Log every post-hook call
    logger.info(f"Post-hook called: step={current_step}, first_completed={first_step_completed}, outputs_count={len(layer_outputs)}")
    
    # Only capture during the first step (step 0)
    if current_step == 0 and not first_step_completed:
        # The output is a tuple (encoder_hidden_states, hidden_states)
        # We want the hidden_states (second element)
        if isinstance(output, tuple) and len(output) == 2:
            hidden_states = output[1]  # Get the hidden_states
            layer_outputs.append(hidden_states.clone().detach())
            logger.debug(f"Cached output hidden states shape: {hidden_states.shape}")

            # Check if we have captured all layers (24 layers for SD3.5)
            if len(layer_outputs) == 24 and len(layer_inputs) == 24:
                logger.info("All 24 layers captured, generating mask...")
                mask_matrix = create_mask_from_differences()
                first_step_completed = True
                logger.info(f"Mask generated successfully! Shape: {mask_matrix.shape if mask_matrix is not None else 'None'}")
                logger.info(f"Mask ready for use in subsequent steps!")
            elif len(layer_outputs) == 24:
                logger.info(f"Outputs captured (24), but inputs only: {len(layer_inputs)}. Waiting for inputs...")

def create_mask_from_differences():
    """
    Create a mask matrix from hidden states differences.
    For each layer, calculate output - input to get the layer's transformation.
    Select the top 20% most changed token positions for each layer.
    Returns:
        mask_matrix: torch.Tensor of shape [num_layers, seq_len], where 1 indicates top 20% change for that layer.
    """
    global layer_inputs, layer_outputs, mask_matrix

    logger.info(f"Creating mask from differences. Inputs: {len(layer_inputs)}, Outputs: {len(layer_outputs)}")
    
    if len(layer_inputs) != len(layer_outputs) or len(layer_inputs) == 0:
        logger.info(f"Not enough data to calculate differences. Inputs: {len(layer_inputs)}, Outputs: {len(layer_outputs)}")
        return None

    num_layers = len(layer_inputs)
    logger.debug(f"Number of layers: {num_layers}")

    # Calculate differences for each layer: output - input
    differences = []
    for i in range(num_layers):
        diff = (layer_outputs[i] - layer_inputs[i]).abs()
        differences.append(diff)
        logger.debug(f"Layer {i} difference shape: {diff.shape}")

    # For each layer difference, extract batch=0 and average over hidden_dim to get [seq_len]
    mean_differences = []
    for i, diff in enumerate(differences):
        # diff: [batch, seq_len, hidden_dim]
        batch_0 = diff[0]  # [seq_len, hidden_dim]
        mean_diff = batch_0.mean(dim=-1)  # [seq_len]
        mean_differences.append(mean_diff)
        logger.debug(f"Layer {i} mean difference shape: {mean_diff.shape}")

    # Stack to get [num_layers, seq_len]
    mean_differences_tensor = torch.stack(mean_differences)  # [num_layers, seq_len]
    num_layers, seq_len = mean_differences_tensor.shape

    # For each row (layer), find top 20% values and set mask
    global_mask_matrix = torch.zeros_like(mean_differences_tensor)
    top_k = max(1, int(0.2 * seq_len))  # At least 1

    for i in range(num_layers):
        row = mean_differences_tensor[i]
        if top_k >= len(row):
            global_mask_matrix[i] = 1.0
        else:
            # Get indices of top 20% values
            topk_indices = torch.topk(row, top_k).indices
            global_mask_matrix[i, topk_indices] = 1.0
        logger.debug(f"Layer {i}: {global_mask_matrix[i].sum().item()} tokens selected as top 20%")

    # Update the global mask_matrix
    mask_matrix = global_mask_matrix
    
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
pre_hooks = []
post_hooks = []
for i, block in enumerate(transformer_8bit.transformer_blocks):
    # Debug: Check block structure
    if hasattr(block.forward, '__code__'):
        logger.info(f"Block forward signature: {block.forward.__code__.co_varnames[:block.forward.__code__.co_argcount]}")
    else:
        logger.info(f"Block forward is a partial object: {block.forward}")
    
    # Register pre-hook for quantization (runs before forward pass)
    pre_hook = block.register_forward_pre_hook(block_pre_hook_fn)
    pre_hooks.append(pre_hook)
    
    # Register post-hook for capturing hidden states (runs after forward pass)
    post_hook = block.register_forward_hook(post_hook_fn)
    post_hooks.append(post_hook)
    
    logger.debug(f"Registered pre-hook and post-hook on transformer block {i}")

# Also try registering hooks on the transformer itself
logger.info("Also registering hooks on transformer level...")
transformer_pre_hook = transformer_8bit.register_forward_pre_hook(transformer_pre_hook_fn)
transformer_post_hook = transformer_8bit.register_forward_hook(transformer_post_hook_fn)

# Debug: Check transformer structure
if hasattr(transformer_8bit.forward, '__code__'):
    logger.info(f"Transformer forward signature: {transformer_8bit.forward.__code__.co_varnames[:transformer_8bit.forward.__code__.co_argcount]}")
else:
    logger.info(f"Transformer forward is a partial object: {transformer_8bit.forward}")

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
    global current_step, first_step_completed
    global subsequent_step_processing, subsequent_step_layer_idx

    # Log current step
    logger.info(f"Starting inference step {current_step}")

    result = original_step(*args, **kwargs)

    # After the step, increment step counter and reset layer index
    current_step += 1
    subsequent_step_processing = False
    subsequent_step_layer_idx = 0
    
    logger.info(f"Completed inference step {current_step-1}, moving to step {current_step}")

    return result

pipeline.scheduler.step = custom_step

prompt = "a tiny astronaut hatching from an egg on the moon"

# Generate image with custom callback
# The mask will be generated automatically during the first timestep
with torch.no_grad():
    image = pipeline(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]

# Store mask for later use (mask was already generated during first timestep)
# if mask_matrix is not None:
#     torch.save(mask_matrix, "attention_mask.pt")
#     logger.info("Mask matrix saved to attention_mask.pt")
# else:
#     logger.warning("No mask matrix was generated!")

# Clean up hooks
for pre_hook in pre_hooks:
    pre_hook.remove()
for post_hook in post_hooks:
    post_hook.remove()

# Clean up transformer-level hooks
transformer_pre_hook.remove()
transformer_post_hook.remove()

# Save the image
image.save("sd3-5.png")