#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
from tqdm import tqdm
from PIL import Image

from grounded_instruct_pix2pix_wrapper import GroundedInstructPixtoPix


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("fork", force=True)

    # -----------------------------
    # Configuration
    # -----------------------------
    device = "cuda:0"
    num_timesteps = 100
    image_guidance_scale = 1.5
    text_guidance_scale = 12.5
    seed = 42
    blending_range = [100, 1]
    verbose = False

    edit_instruction = "Turn the table pink"
    

    input_folder = "/home/ubuntu/spjain/Grounded-Instruct-Pix2Pix/src/garden"
    output_folder = "/home/ubuntu/spjain/Grounded-Instruct-Pix2Pix/src/edited_garden_pink_table_qwen"
    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # Initialize model wrapper
    # -----------------------------
    editor = GroundedInstructPixtoPix(
        num_timesteps=num_timesteps,
        device=device,
        image_guidance_scale=image_guidance_scale,
        text_guidance_scale=text_guidance_scale,
        start_blending_at_tstep=blending_range[0],
        end_blending_at_tstep=blending_range[1],
        prompt=edit_instruction,
        seed=seed,
        verbose=verbose,
        debug = True
    )

    # -----------------------------
    # Collect images
    # -----------------------------
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    print(f"Found {len(image_paths)} images.")

    # -----------------------------
    # Batch editing
    # -----------------------------
    for img_path in tqdm(image_paths, desc="Editing Images"):

        # Load and resize
        img_pil = editor.load_pil_image(img_path)

        # Run editing
        edited_pil = editor.edit_image(img_pil, img_path)

        # Save result
        save_path = os.path.join(output_folder, os.path.basename(img_path))
        edited_pil.resize(img_pil.size).save(save_path)

    print("✨ Batch editing completed!")
