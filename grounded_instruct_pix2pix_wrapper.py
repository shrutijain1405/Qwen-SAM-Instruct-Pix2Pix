import math
import torch

from tqdm.notebook import tqdm
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from diffusers import DDIMScheduler, DDIMInverseScheduler
from external_mask_extractor_qwen import ExternalMaskExtractor
from pipeline_stable_diffusion_grounded_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline

from transformers import logging
import random
logging.set_verbosity_error()

class GroundedInstructPixtoPix():
    def __init__(self, num_timesteps: int = 100, device: str = 'cuda:0', image_guidance_scale: float = 1.5, 
                 text_guidance_scale: float = 7.5, start_blending_at_tstep: int = 100,
                 end_blending_at_tstep: int = 1, prompt: str = '', seed: int = 42, 
                 verbose: bool = False, debug: bool = False):
        
        self.num_timesteps = num_timesteps
        self.device = device
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale
        self.prompt = prompt
        self.seed = seed
        self.verbose = verbose
        self.blending_range = [start_blending_at_tstep,end_blending_at_tstep]
        self.mask_extractor = ExternalMaskExtractor(device=self.device, debug = debug)
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                                torch_dtype=torch.float16,
                                                                                safety_checker=None).to(self.device)
        self.pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config, set_alpha_to_zero=False) #noising

        self.pipeline.scheduler.set_timesteps(self.num_timesteps) #denoising
        self.pipeline.inverse_scheduler.set_timesteps(self.num_timesteps)


    def load_pil_image(self, image_path, resolution=512):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        return image



    def edit_image(self, image, img_path):
        # to_pil = transforms.ToPILImage()
        # image = to_pil(image)
        external_mask_pil = self.mask_extractor.get_external_mask(image, img_path, self.prompt, verbose=self.verbose)
        inv_results = self.pipeline.invert(self.prompt, image, num_inference_steps=self.num_timesteps, inv_range=self.blending_range) #noising
        generator = torch.Generator(self.device).manual_seed(self.seed) if self.seed is not None else torch.Generator(self.device)
        edited_image = self.pipeline(self.prompt, src_mask=external_mask_pil, image=image,
                                guidance_scale=self.text_guidance_scale, image_guidance_scale=self.image_guidance_scale,
                                num_inference_steps=self.num_timesteps, generator=generator).images[0] #denoising
        # to_tensor = transforms.ToTensor()
        # edited_image = to_tensor(edited_image)
        return edited_image