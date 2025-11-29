import math
import torch

from tqdm.notebook import tqdm
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from diffusers import DDIMScheduler, DDIMInverseScheduler
from Grounded_Instruct_Pix2Pix.external_mask_extractor import ExternalMaskExtractor
from Grounded_Instruct_Pix2Pix.pipeline_stable_diffusion_grounded_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline

from transformers import logging
import random
logging.set_verbosity_error()

class GroundedInstructPixtoPix():
    def __init__(self, num_timesteps: int = 100, device: str = 'cuda:0', image_guidance_scale: float = 1.5, 
                 text_guidance_scale: float = 7.5, denoising_steps: int = 20, prompt: str = '', seed: int = 42, 
                 verbose: bool = False, idu: bool = False):
        
        self.num_timesteps = num_timesteps
        self.idu = idu
        self.device = device
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale
        self.denoising_steps = denoising_steps
        self.prompt = prompt
        self.seed = seed
        self.verbose = verbose
        self.mask_extractor = ExternalMaskExtractor(device=self.device)
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                                torch_dtype=torch.float16,
                                                                                safety_checker=None).to(self.device)
        self.pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipeline.scheduler.config, set_alpha_to_zero=False) #noising

        self.pipeline.scheduler.set_timesteps(self.num_timesteps) #denoising
        self.pipeline.inverse_scheduler.set_timesteps(self.num_timesteps)


    def load_pil_image(image_path, resolution=512):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        return image



    def edit_image(self, image):
        to_pil = transforms.ToPILImage()
        image_pil = to_pil(image)
        external_mask_pil, chosen_noun_phrase, clip_scores_dict = self.mask_extractor.get_external_mask(image_pil, self.prompt, verbose=self.verbose)
        if(self.idu):
            start_t = random.randint(int(0.7*self.num_timesteps), int(0.98*self.num_timesteps))
        else:
            start_t = self.num_timesteps
        blending_range = [start_t,1]
        inv_results = self.pipeline.invert(self.prompt, image_pil, num_inference_steps=start_t, inv_range=blending_range) #noising
        if(self.idu):
            img_inp = inv_results.latents
        else:
            img_inp = image_pil
        generator = torch.Generator(self.device).manual_seed(self.seed) if self.seed is not None else torch.Generator(self.device)
        edited_image = self.pipeline(self.prompt, src_mask=external_mask_pil, image=img_inp,
                                guidance_scale=self.text_guidance_scale, image_guidance_scale=self.image_guidance_scale,
                                num_inference_steps=self.denoising_steps, generator=generator).images[0] #denoising
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(edited_image)
        return tensor_image