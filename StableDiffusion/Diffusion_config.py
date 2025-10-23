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
Hidden_MASK = False
Context_MASK = False

# Global variables for timestep skipping state management
GLOBAL_TIMESTEP_SKIP_STATE = {
    'TS_STATE': [], # 0 is normal computation, 1 is skip with PSI, 2 is skip with reuse
    'noise_pred_cache': [],  # Store noise_pred from previous timesteps
    'cache_step': [],        # Store step index from previous timesteps
    'current_step': 0,
    'skip_steps': [2, 44, 3, 5],  # Number of initial steps to keep normal computation
    'total_steps': 50,          # Total number of inference steps
    'skipped_timesteps': [],     # Store indices of skipped timesteps
    'all_noise_pred_cache': [],        # Store all noise_pred from previous timesteps
    'all_cache_step': [],        # Store all cache_step from previous timesteps
    'mask_cache': [],        # Store mask from previous timesteps
    'context_mask_cache': [],        # Store context mask from previous timesteps
}


def generate_ts_state():
    """
    Generate the complete TS_STATE list based on skip_steps configuration
    """
    global GLOBAL_TIMESTEP_SKIP_STATE
    
    total_steps = GLOBAL_TIMESTEP_SKIP_STATE['total_steps']
    skip_steps = GLOBAL_TIMESTEP_SKIP_STATE['skip_steps']
    
    start_step = skip_steps[0]  # optimization start
    end_step = skip_steps[1]    # optimization end
    loop_length = skip_steps[2] # loop length (must be >3)
    
    # Initialize all steps as normal computation (0)
    ts_state = [0] * total_steps
    
    # Generate loop pattern: 1, 2, 2, 0 (for loop=4) or 1, 2, 0 (for loop=3)
    loop_pattern = [1]  # Always start with 1
    # loop_pattern.append(2)  # Then 2
    
    # Add additional 2s based on loop length
    for i in range(loop_length - 2):
        loop_pattern.append(2)
    
    loop_pattern.append(0)  # End with 0
    
    # Apply the pattern from start_step to end_step
    pattern_index = 0
    for step in range(start_step, min(end_step + 1, total_steps)):
        ts_state[step] = loop_pattern[pattern_index % len(loop_pattern)]
        pattern_index += 1
    if (end_step+1) < total_steps:
        for i in range(end_step+1, total_steps):
            ts_state[i] = 0
    # Update the global state
    GLOBAL_TIMESTEP_SKIP_STATE['TS_STATE'] = ts_state
    
    # Log the generated pattern for verification
    logger.info(f"Generated TS_STATE: {ts_state}")
    logger.info(f"Total skipped timesteps: {ts_state.count(0)}")
    logger.info(f"Pattern applied from step {start_step} to {end_step} with loop length {loop_length}")
    logger.info(f"Loop pattern: {loop_pattern}")

def get_skipped_noise_pred(func, step_index):

    global GLOBAL_TIMESTEP_SKIP_STATE
    # pdb.set_trace()
    cache = GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache']
    cache_step = GLOBAL_TIMESTEP_SKIP_STATE['cache_step']

    if func == 2:  # Skip, use reuse
        print(f"step_index: {step_index}, use reuse with  {cache_step[-2]}")
        return cache[-2]
        # print(f"step_index: {step_index}, use psi with  2*{cache_step[-1]} - {cache_step[-2]}")
        # return 2 * cache[-1] - cache[-2]
    elif func == 1:  # Skip, use linear interpolation
        print(f"step_index: {step_index}, use psi with  2*{cache_step[-1]} - {cache_step[-2]}")
        return 2 * cache[-1] - cache[-2]
        
    else:
        raise ValueError(f"Invalid skip pattern for step {step_index}, cycle position {cycle_position}")

def cache_noise_pred(noise_pred, step_index, keep_all=False):
    """Cache noise_pred for future interpolation/reuse"""
    global GLOBAL_TIMESTEP_SKIP_STATE
    
    # Keep only the last 2 values to save memory
    cache = GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache']
    cache_step = GLOBAL_TIMESTEP_SKIP_STATE['cache_step']
    cache.append(noise_pred.clone().detach())
    cache_step.append(step_index)
    if keep_all:
        all_cache = GLOBAL_TIMESTEP_SKIP_STATE['all_noise_pred_cache']
        all_cache_step = GLOBAL_TIMESTEP_SKIP_STATE['all_cache_step']
        all_cache.append(noise_pred.clone().detach())
        all_cache_step.append(step_index)
        GLOBAL_TIMESTEP_SKIP_STATE['all_noise_pred_cache'] = all_cache
        GLOBAL_TIMESTEP_SKIP_STATE['all_cache_step'] = all_cache_step
    
    if len(cache) > 2:
        cache.pop(0)  # Remove oldest value
        cache_step.pop(0)
    
    GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache'] = cache
    GLOBAL_TIMESTEP_SKIP_STATE['cache_step'] = cache_step

def record_skipped_timestep(step_index):
    """Record the index of a skipped timestep"""
    global GLOBAL_TIMESTEP_SKIP_STATE
    
def update_mask(top_k):
    """Get mask for outliers in noise_pred"""
    global GLOBAL_TIMESTEP_SKIP_STATE
    cache = GLOBAL_TIMESTEP_SKIP_STATE['noise_pred_cache']
    
    if len(cache) == 0:
        print("Warning: cache is empty, returning zero mask")
        return torch.zeros(1, 16, 128, 128)
    
    # 获取最新的 noise_pred
    noise_pred = cache[-1] # [2, 4096, 1536]

    # Compute mean absolute value over feature dimension (dim=1), result: [4096]
    meanabs_1d = noise_pred[0].abs().mean(dim=1)
    topk_num = max(1, int(top_k * meanabs_1d.numel())) 
    topk_indices = torch.topk(meanabs_1d, topk_num).indices

    GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'] = topk_indices


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


# 帮我在这个文件里新建一个类，应该是个字典，用来存一个timestep内网络中特定层的pt数据，结构差不多长这样：
# {'JA-toq': [], 'JA-tok': [], 'JA-tov': [], 'DA-toq': [], 'DA-tok': [], 'DA-tov': [], 'end': []}
class TimestepData:
    def __init__(self):
        # Store lists for each layer type
        self.JA_toq = []
        self.JA_tok = []
        self.JA_tov = []
        self.JA_toq_context = []
        self.JA_tok_context = []
        self.JA_tov_context = []
        self.DA_toq = []
        self.DA_tok = []
        self.DA_tov = []
        self.end = []
        self.end_context = []

    def append(self, layer_name, data):
        # Append data to the correct attribute list by name
        if hasattr(self, layer_name):
            getattr(self, layer_name).append(data)
        else:
            raise KeyError(f"Layer name '{layer_name}' not found in TimestepData.")

GLOBAL_TIMESTEP_DATA = TimestepData()
GLOBAL_GETMASK_LAYER = [] #[0, 5, 10, 15, 20]


AllTSDataList = []
LOOP_Layer = GLOBAL_TIMESTEP_SKIP_STATE['skip_steps'][3]
TOPK_RATIO = 1

def generate_GLOBAL_GETMASK_LAYER():
    global GLOBAL_GETMASK_LAYER
    loop_num = GLOBAL_TIMESTEP_SKIP_STATE['skip_steps'][3]
    for i in range(24):
        if i % loop_num == 0:
            GLOBAL_GETMASK_LAYER.append(i)


def cache_GLOBAL_TIMESTEP_DATA(data):
    global GLOBAL_TIMESTEP_DATA
    global AllTSDataList
    AllTSDataList.append(GLOBAL_TIMESTEP_DATA)
    if len(AllTSDataList) > 3:
        AllTSDataList.pop(0)
    GLOBAL_TIMESTEP_DATA = TimestepData()

def update_mask_inTS(top_k):
    # clean the mask cache
    GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'] = []
    GLOBAL_TIMESTEP_SKIP_STATE['context_mask_cache'] = []

    global AllTSDataList
    LastTSData = AllTSDataList[-1]
    for layer_id in GLOBAL_GETMASK_LAYER:
        # end_data = LastTSData.end[layer_id]
        end_data = LastTSData.end[layer_id]
        # generate mask for image
        meanabs_1d = end_data[0].abs().mean(dim=1)
        topk_num = max(1, int(top_k * meanabs_1d.numel())) 
        topk_indices = torch.topk(meanabs_1d, topk_num).indices
        GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'].append(topk_indices)
        # generate mask for context
        # pdb.set_trace()
        end_context_data = LastTSData.end_context[layer_id]
        meanabs_1d = end_context_data[0].abs().mean(dim=1)
        topk_num = max(1, int(top_k * meanabs_1d.numel())) 
        topk_indices = torch.topk(meanabs_1d, topk_num).indices
        GLOBAL_TIMESTEP_SKIP_STATE['context_mask_cache'].append(topk_indices)



def update_Diffmask_inTS(top_k, current_TS):
    # clean the mask cache
    GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'] = []
    GLOBAL_TIMESTEP_SKIP_STATE['context_mask_cache'] = []

    global AllTSDataList
    LastTSData = AllTSDataList[-1]
    if current_TS <= GLOBAL_TIMESTEP_SKIP_STATE['skip_steps'][1]:
        for layer_id in GLOBAL_GETMASK_LAYER:
            # end_data = LastTSData.end[layer_id]
            end_data = LastTSData.end[layer_id]
            # generate mask for image
            meanabs_1d = end_data[0].abs().mean(dim=1)
            topk_num = max(1, int(top_k * meanabs_1d.numel())) 
            topk_indices = torch.topk(meanabs_1d, topk_num).indices
            GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'].append(topk_indices)
            # generate mask for context
            # pdb.set_trace()
            end_context_data = LastTSData.end_context[layer_id]
            meanabs_1d = end_context_data[0].abs().mean(dim=1)
            topk_num = max(1, int(top_k * meanabs_1d.numel()))
            topk_indices = torch.topk(meanabs_1d, topk_num).indices
            GLOBAL_TIMESTEP_SKIP_STATE['context_mask_cache'].append(topk_indices)
    else:
        PSITSData = AllTSDataList[-3]
        ReuseTSData = AllTSDataList[-2]
        for layer_id in GLOBAL_GETMASK_LAYER:
            # PredTSData = 2*ReuseTSData.end[layer_id]-PSITSData.end[layer_id]
            DiffTSData = LastTSData.end[layer_id]-ReuseTSData.end[layer_id]
            meanabs_1d = DiffTSData[0].abs().mean(dim=1)
            topk_num = max(1, int(top_k * meanabs_1d.numel())) 
            topk_indices = torch.topk(meanabs_1d, topk_num).indices
            GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'].append(topk_indices)

            # PredTSData = 2*ReuseTSData.end_context[layer_id]-PSITSData.end_context[layer_id]
            DiffTSData = LastTSData.end_context[layer_id]-ReuseTSData.end_context[layer_id]
            meanabs_1d = DiffTSData[0].abs().mean(dim=1)
            topk_num = max(1, int(top_k * meanabs_1d.numel())) 
            topk_indices = torch.topk(meanabs_1d, topk_num).indices
            GLOBAL_TIMESTEP_SKIP_STATE['context_mask_cache'].append(topk_indices)