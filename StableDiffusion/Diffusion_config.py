# SD3.5 Global Variable Based Selective Quantization Implementation
# This code implements the following workflow:
# 1. In step 0: Capture input hidden_states of each JointTransformerBlock via global variables
# 2. Calculate difference values between adjacent layers and generate mask matrix
# 3. In subsequent steps: Use mask matrix to selectively quantize JointTransformerBlock inputs

import imp
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

def setup_model_and_pipeline_no_quant():
    """Setup model and pipeline without quantization for evaluation"""
    # Setup text encoder without quantization
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )

    # Setup transformer without quantization
    transformer = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        subfolder="transformer",
        torch_dtype=torch.float16,
    )

    # Create pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        text_encoder=text_encoder,
        transformer=transformer,
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


# 这里能控制网络里对哪些计算层进行优化，目前就是toqkv以及Attention里的lienarout以及后续的MLP
# 注释掉这里面的层可以控制对哪些层进行优化（同时还需要把库代码里的对应代码也注释掉）
DATA_TYPE_NAMES = [
    'JA_toq', 'JA_tok', 'JA_tov', #'JA_sv',
    'JA_toq_context', 'JA_tok_context', 'JA_tov_context',
    'DA_toq', 'DA_tok', 'DA_tov', #'DA_sv',
    'JA_out', 'DA_out', # the output linear of attention
    'MLP_up_out', 'MLP_down_out',
    # 'MLP_up_out_context', 'MLP_down_out_context',
    # 'Layer_out', 
    # 'Layer_out_context'
]

class TimestepData:
    def __init__(self):
        for data_type in DATA_TYPE_NAMES:
            setattr(self, data_type, [])
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

def update_mask_inTS_based_on_end_data(top_k):
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


def detect_outliers_zscore_layer_adaptive(data, layer_id, top_k,total_layers=24):
    """
    使用分层Z-Score方法检测outliers
    根据层深度调整阈值，确保每层都有合理的token选择
    """
    # 计算Z-Score
    mean = data.mean(dim=-1, keepdim=True)
    std = data.std(dim=-1, keepdim=True)
    z_scores = torch.abs((data - mean) / (std + 1e-8))
    
    # 计算每个token的最大Z-Score
    max_z_scores = z_scores.max(dim=-1)[0]  # [batch, seq_len]
    
    # 根据层深度调整阈值策略
    if layer_id < total_layers // 3:  # 浅层 (0-9)
        # 使用更宽松的阈值，选择更多outliers
        threshold_percentile = top_k  
    elif layer_id < 2 * total_layers // 3:  # 中层 (10-19)
        # 使用中等阈值
        threshold_percentile = top_k
    else:  # 深层 (20-27)
        # 使用更严格的阈值，选择更少outliers
        threshold_percentile = top_k
    
    # 根据百分位数计算阈值
    # 确保张量是float类型，因为torch.quantile()要求float或double类型
    max_z_scores_flat = max_z_scores.flatten().float()
    threshold = torch.quantile(max_z_scores_flat, threshold_percentile)
    
    # 创建outlier mask
    outlier_mask = max_z_scores > threshold
    
    return max_z_scores, outlier_mask, threshold, threshold_percentile


def update_mask_inTS_for_all_data_types_zscore(top_k=None, use_zscore=True, enable_hidden_dim_outlier=False):
    """
    基于分层Z-Score方法为TimestepData中每个name生成各自的mask
    并将mask以TimestepData的形式存储在GLOBAL_TIMESTEP_SKIP_STATE的'mask_cache'中
    
    Args:
        top_k: 保留的token比例（当use_zscore=False时使用）
        use_zscore: 是否使用Z-Score方法，否则使用原来的top-k方法
        enable_hidden_dim_outlier: 是否同时在hidden_dim维度上检测outlier
    """
    global AllTSDataList, GLOBAL_TIMESTEP_SKIP_STATE
    
    # 清理旧的mask cache
    GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'] = TimestepData()
    
    LastTSData = AllTSDataList[-1]
    # 为每个数据类型生成mask
    for data_type in DATA_TYPE_NAMES:
        # 为每个layer生成mask
        for layer_id in GLOBAL_GETMASK_LAYER:
            # import pdb; pdb.set_trace()
            # pdb.set_trace()
            # 获取该数据类型在指定层的数据
            if hasattr(LastTSData, data_type) and len(getattr(LastTSData, data_type)) > layer_id:
                data = getattr(LastTSData, data_type)[layer_id]
                
                if len(data.shape) >= 2:  # 确保数据有足够的维度
                    if use_zscore:
                        # 使用分层Z-Score方法
                        outlier_scores, outlier_mask, threshold, percentile = detect_outliers_zscore_layer_adaptive(
                            data, layer_id, (1-top_k)
                        )
                        keep_indices = torch.where(outlier_mask[0])[0]
                        # print(keep_indices)
                    else:
                        # 使用原来的top-k方法
                        meanabs_1d = data[0].abs().mean(dim=1) if len(data.shape) > 2 else data.abs().mean(dim=1)
                        
                        if top_k == 0:
                            keep_indices = torch.tensor([], dtype=torch.long, device=meanabs_1d.device)
                            keep_indices_hidden = torch.tensor([], dtype=torch.long, device=meanabs_1d.device)
                        else:
                            # Token维度的outlier检测
                            try:
                                topk_num = max(1, int(top_k * meanabs_1d.numel()))
                                keep_indices = torch.topk(meanabs_1d, topk_num, largest=False).indices
                            except Exception as e:
                                print(f"Error in topk: {e}")
                                import pdb; pdb.set_trace()
                            
                            # Hidden维度的outlier检测（如果启用）
                            if enable_hidden_dim_outlier and len(data.shape) > 2 and top_k > 0:
                                # 在hidden_dim维度上计算mean absolute value
                                meanabs_hidden = data[0].abs().mean(dim=0)  # [hidden_dim]
                                topk_num_hidden = max(1, int(top_k * meanabs_hidden.numel()))
                                keep_indices_hidden = torch.topk(meanabs_hidden, topk_num_hidden, largest=False).indices
                            else:
                                keep_indices_hidden = torch.tensor([], dtype=torch.long, device=meanabs_1d.device)
                    
                    # 将mask添加到对应数据类型的列表中
                    if enable_hidden_dim_outlier and len(keep_indices_hidden) > 0:
                        # 存储[keep_indices, keep_indices_hidden]格式
                        mask_data = [keep_indices, keep_indices_hidden]
                    else:
                        # 只存储keep_indices
                        mask_data = keep_indices
                    
                    getattr(GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'], data_type).append(mask_data)
                else:
                    # 如果数据维度不够，创建空mask
                    print(f"Data type {data_type} has insufficient dimensions for layer {layer_id}")
                    getattr(GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'], data_type).append(torch.tensor([], dtype=torch.long))
            else:
                # 如果数据不存在，创建空mask
                if layer_id < 13 and not 'DA' in data_type:
                    print(f"Data type {data_type} has no data for layer {layer_id}")
                    getattr(GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'], data_type).append(torch.tensor([], dtype=torch.long))

def update_mask_inTS_for_all_data_types_diffvalue(top_k=None, use_zscore=True, enable_hidden_dim_outlier=False):
    """
    基于分层Z-Score方法为TimestepData中每个name生成各自的mask
    并将mask以TimestepData的形式存储在GLOBAL_TIMESTEP_SKIP_STATE的'mask_cache'中
    
    Args:
        top_k: 保留的token比例（当use_zscore=False时使用）
        use_zscore: 是否使用Z-Score方法，否则使用原来的top-k方法
        enable_hidden_dim_outlier: 是否同时在hidden_dim维度上检测outlier
    """
    global AllTSDataList, GLOBAL_TIMESTEP_SKIP_STATE
    
    # 清理旧的mask cache
    GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'] = TimestepData()
    
    ReuseTSData = AllTSDataList[-2]
    LastTSData = AllTSDataList[-1]
    # 为每个数据类型生成mask
    for data_type in DATA_TYPE_NAMES:
        for layer_id in GLOBAL_GETMASK_LAYER:
            if hasattr(LastTSData, data_type) and len(getattr(LastTSData, data_type)) > layer_id:
                data = getattr(LastTSData, data_type)[layer_id]
                Reuse_data = getattr(ReuseTSData, data_type)[layer_id]
                Diff_data = (data - Reuse_data)
                if len(data.shape) >= 2:  # 确保数据有足够的维度
                    if use_zscore:
                        # 使用分层Z-Score方法
                        outlier_scores, outlier_mask, threshold, percentile = detect_outliers_zscore_layer_adaptive(
                            Diff_data, layer_id, (1-top_k)
                        )
                        keep_indices = torch.where(outlier_mask[0])[0]
                        # print(keep_indices)
                    else:
                        # 使用原来的top-k方法
                        meanabs_1d = Diff_data[0].abs().mean(dim=1) if len(Diff_data.shape) > 2 else Diff_data.abs().mean(dim=1)
                        
                        if top_k == 0:
                            keep_indices = torch.tensor([], dtype=torch.long, device=meanabs_1d.device)
                            keep_indices_hidden = torch.tensor([], dtype=torch.long, device=meanabs_1d.device)
                        else:
                            # Token维度的outlier检测
                            try:
                                topk_num = max(1, int(top_k * meanabs_1d.numel()))
                                keep_indices = torch.topk(meanabs_1d, topk_num, largest=False).indices
                            except Exception as e:
                                print(f"Error in topk: {e}")
                                import pdb; pdb.set_trace()

                            # Hidden维度的outlier检测（如果启用）
                            if enable_hidden_dim_outlier and len(Diff_data.shape) > 2 and top_k > 0:
                                # 在hidden_dim维度上计算mean absolute value
                                meanabs_hidden = Diff_data[0].abs().mean(dim=0)  # [hidden_dim]
                                topk_num_hidden = max(1, int(top_k * meanabs_hidden.numel()))
                                keep_indices_hidden = torch.topk(meanabs_hidden, topk_num_hidden, largest=False).indices
                            else:
                                keep_indices_hidden = torch.tensor([], dtype=torch.long, device=meanabs_1d.device)
                    
                    # 将mask添加到对应数据类型的列表中
                    if enable_hidden_dim_outlier and len(keep_indices_hidden) > 0:
                        # 存储[keep_indices, keep_indices_hidden]格式
                        mask_data = [keep_indices, keep_indices_hidden]
                    else:
                        # 只存储keep_indices
                        mask_data = keep_indices
                    
                    getattr(GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'], data_type).append(mask_data)
                else:
                    # 如果数据维度不够，创建空mask
                    print(f"Data type {data_type} has insufficient dimensions for layer {layer_id}")
                    getattr(GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'], data_type).append(torch.tensor([], dtype=torch.long))
            else:
                # 如果数据不存在，创建空mask
                if layer_id < 13 and not 'DA' in data_type:
                    print(f"Data type {data_type} has no data for layer {layer_id}")
                    getattr(GLOBAL_TIMESTEP_SKIP_STATE['mask_cache'], data_type).append(torch.tensor([], dtype=torch.long))



def EchoFlow_opti_with_stored_mask(current_tensor, data_type, mode, current_layer):
    """
    使用存储在TimestepData中的mask进行EchoFlow优化
    
    Args:
        current_tensor: The current tensor to be optimized (query/key/value)
        data_type: Type of data ('JA_toq', 'JA_tok', 'JA_tov', 'JA_toq_context', 'JA_tok_context', 'JA_tov_context', 
                  'DA_toq', 'DA_tok', 'DA_tov')
        mode: Optimization mode (1 for PSI interpolation, 2 for reuse)
        current_layer: Current layer index
    
    Returns:
        Optimized tensor
    """
    global GLOBAL_TIMESTEP_SKIP_STATE, AllTSDataList
    
    if mode == 1:  # PSI interpolation
        psi_tensor = 2 * getattr(AllTSDataList[-1], data_type)[current_layer] - getattr(AllTSDataList[-2], data_type)[current_layer]
    elif mode == 2:  # Reuse
        psi_tensor = getattr(AllTSDataList[-2], data_type)[current_layer]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 1 (PSI) or 2 (reuse)")
    # import pdb; pdb.set_trace()
    # 从TimestepData格式的mask_cache中获取mask
    mask_cache = GLOBAL_TIMESTEP_SKIP_STATE['mask_cache']
    if hasattr(mask_cache, data_type) and len(getattr(mask_cache, data_type)) > current_layer:
        mask_data = getattr(mask_cache, data_type)[current_layer]
        
        # 检查mask_data的格式
        if isinstance(mask_data, list) and len(mask_data) == 2:
            # 新格式：[keep_indices, keep_indices_hidden]
            keep_indices, keep_indices_hidden = mask_data
        else:
            # 旧格式：只有keep_indices
            keep_indices = mask_data
            keep_indices_hidden = torch.tensor([], dtype=torch.long, device=current_tensor.device)
        
        if len(keep_indices) > 0:
            # 创建token维度的mask
            full_mask = torch.zeros(current_tensor.shape[1], dtype=torch.bool, device=current_tensor.device)
            full_mask[keep_indices] = True
            full_mask = full_mask.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            # 如果有hidden维度的mask，扩展到与current_tensor相同的尺寸
            if len(keep_indices_hidden) > 0 and len(current_tensor.shape) > 2:
                # 将mask扩展到[1, seq_len, hidden_dim]
                full_mask = full_mask.expand(-1, -1, current_tensor.shape[-1]).clone()  # [1, seq_len, hidden_dim]
                # 在hidden_dim维度上设置对应的位置为True
                full_mask[:, :, keep_indices_hidden] = True

            return torch.where(full_mask, current_tensor, psi_tensor)
    
    return psi_tensor
