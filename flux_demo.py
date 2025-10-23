import argparse
import logging
import math
import time
import torch
import lpips

import sys
import os



# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from torchvision.utils import save_image
from zeus import patch

# Add local diffusers directory to Python path
local_diffusers_path = '/home/yf184/diffusers/src'
if local_diffusers_path not in sys.path:
    sys.path.insert(0, local_diffusers_path)

from diffusers import FluxPipeline
import torchvision.transforms as T


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Cinematic photograph of a dark cat walking in the fantasy moonlight garden")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    args = parser.parse_args()

    lora_path = ""
    seed = args.seed
    prompt = args.prompt

    # baseline_pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to('cuda')

    # Warmup GPU. Only for testing the speed.
    # logging.info("Warming up GPU...")
    # for _ in range(1):
    #     set_random_seed(seed)
    #     _ = baseline_pipe(
    #         prompt,
    #         height=args.height,
    #         width=args.width,
    #         guidance_scale=3.5,
    #         num_inference_steps=50,
    #         max_sequence_length=512,
    #         output_type='pt'
    #     ).images

    # # Baseline
    # logging.info("Running baseline...")
    # start_time = time.time()
    # set_random_seed(seed)

    # ori_output = baseline_pipe(
    #     prompt,
    #     height=args.height,
    #     width=args.width,
    #     guidance_scale=3.5,
    #     num_inference_steps=50,
    #     max_sequence_length=512,
    #     output_type='pt'
    # ).images
    # baseline_use_time = time.time() - start_time
    # logging.info("Baseline: {:.2f} seconds".format(baseline_use_time))

    # del baseline_pipe
    # torch.cuda.empty_cache()

    # Zeus
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to('cuda')
    # pipe.load_lora_weights(lora_path)

    patch.apply_patch(pipe,
                      acc_range=(10, 45),

                      interp_mode="psi",
                      caching_mode="reuse_interp",
                      denominator=3,
                      modular=(0,1,),

                      lagrange_int=6,
                      lagrange_step=24,
                      lagrange_term=3)

    # Warmup GPU. Only for testing the speed.
    # logging.info("Warming up GPU...")
    # for _ in range(1):
    #     set_random_seed(seed)
    #     _ = pipe(
    #         prompt,
    #         height=args.height,
    #         width=args.width,
    #         guidance_scale=3.5,
    #         num_inference_steps=50,
    #         max_sequence_length=512,
    #         output_type='pt'
    #     ).images
    #     patch.reset_cache(pipe)

    logging.info("Running ⚡Zeus...")
    set_random_seed(seed)
    start_time = time.time()

    cap_output = pipe(
        prompt,
        height=args.height,
        width=args.width,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        output_type='pt'
    ).images
    use_time = time.time() - start_time
    logging.info("⚡Zeus: {:.2f} seconds".format(use_time))

    print(pipe.transformer._cache_bus.skipping_path)

    logging.info("Baseline: {:.2f} seconds. CAP: {:.2f} seconds".format(50, use_time))
    # logging.info("Baseline: {:.2f} seconds. CAP: {:.2f} seconds".format(baseline_use_time, use_time))

    # save_image([ori_output[0], cap_output[0]], "output.png")
    # logging.info("Saved to output.png. Done!")

    # print("Evaluating LPIPS")
    # p_r = torch.stack([T.Compose([
    #     T.Normalize((0.5,), (0.5,))
    # ])(img) for img in ori_output]).to('cuda')

    # p_o = torch.stack([T.Compose([
    #     T.Normalize((0.5,), (0.5,))
    # ])(img) for img in cap_output]).to('cuda')

    # loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
    # d = loss_fn_alex(p_r, p_o)
    # print(f"LPIPS: {d.item()}")