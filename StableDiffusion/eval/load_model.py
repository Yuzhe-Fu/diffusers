import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,  # switch to float16 if your GPU lacks bf16
    variant="fp16"               # omit if you change the dtype
).to("cuda:2")                   # match your CUDA device

image = pipe("a translucent glass vase full of tulips").images[0]
image.save("vase_demo/sample.png")
