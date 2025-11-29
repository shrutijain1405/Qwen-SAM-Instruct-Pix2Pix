#!/usr/bin/env python
# coding: utf-8

import os
import math
import torch
from glob import glob

from tqdm import tqdm
from PIL import Image, ImageOps

from diffusers import DDIMScheduler, DDIMInverseScheduler
from external_mask_extractor import ExternalMaskExtractor
from pipeline_stable_diffusion_grounded_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline

from transformers import logging
logging.set_verbosity_error()


# ---------------------------------------------------------
# Utility: Load + Resize PIL Image
# ---------------------------------------------------------
def load_pil_image(image_path, resolution=512):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    return image


# ---------------------------------------------------------
# Setup Mask Extractor + Pipeline
# ---------------------------------------------------------
device = "cuda:0"
mask_extractor = ExternalMaskExtractor(device=device)

num_timesteps = 100
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(
    pipeline.scheduler.config, set_alpha_to_zero=False
)

pipeline.scheduler.set_timesteps(num_timesteps)
pipeline.inverse_scheduler.set_timesteps(num_timesteps)


# ---------------------------------------------------------
# Single Image Inference
# ---------------------------------------------------------
def inference(pipeline, image_pil, instruction,
              image_guidance_scale, text_guidance_scale,
              seed, blending_range, verbose=False):

    external_mask_pil, chosen_noun_phrase, clip_scores_dict = mask_extractor.get_external_mask(image_pil, instruction, verbose=verbose)

    _ = pipeline.invert(instruction, image_pil,
                        num_inference_steps=num_timesteps,
                        inv_range=blending_range)

    generator = (torch.Generator(device).manual_seed(seed)
                 if seed is not None else torch.Generator(device))

    edited_image = pipeline(
        instruction,
        src_mask=external_mask_pil,
        image=image_pil,
        guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_timesteps,
        generator=generator
    ).images[0]

    return edited_image


# ---------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------
input_folder = "/home/ubuntu/spjain/Grounded-Instruct-Pix2Pix/src/garden"
output_folder = "/home/ubuntu/spjain/Grounded-Instruct-Pix2Pix/src/edited_garden_roses_centre"

os.makedirs(output_folder, exist_ok=True)

# Your edit prompt
edit_instruction = "turn the fake flowers in the centre into red roses"

image_guidance_scale = 1.5
guidance_scale = 7.5
seed = 42

start_blending_at_tstep = 100
end_blending_at_tstep = 1
blending_range = [start_blending_at_tstep, end_blending_at_tstep]

verbose = False

# Collect all images in folder
image_paths = []
for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
    image_paths.extend(glob(os.path.join(input_folder, ext)))

print(f"Found {len(image_paths)} images.")

# Process each image
for img_path in tqdm(image_paths, desc="Editing Images"):
    img = load_pil_image(img_path)

    edited = inference(
        pipeline, img, edit_instruction,
        image_guidance_scale, guidance_scale,
        seed, blending_range, verbose
    )

    # Save output using same filename
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_folder, filename)

    edited.resize(img.size).save(save_path)

print("✨ Batch editing completed!")
