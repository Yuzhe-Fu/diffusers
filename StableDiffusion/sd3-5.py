# set the cuda device
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, CLIPTextModelWithProjection
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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

pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    # device_map="balanced",
)
# pipeline = pipeline.to("cuda")
pipeline.enable_model_cpu_offload()
prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
#save the image
image.save("sd3-5-new.png")