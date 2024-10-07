import os
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from models.diffusion import Model
from functions.process_data import *

from omegaconf import OmegaConf 
from ldm.util import instantiate_from_config
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def load_model_from_config(config, ckpt, device): 
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd 

    model = instantiate_from_config(config.model)
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")
        
    model.to(device).float() 
    model.eval() 
    return model 


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float32, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # Stable Diffusion config
        config_path = "stable_diffusion/768-v-ema.yaml"
        self.config_sd = OmegaConf.load(config_path)

        self.num_timesteps = self.config_sd.model.params.timesteps
        betas = self.get_beta_schedule(
            beta_start=self.config_sd.model.params.linear_start,
            beta_end=self.config_sd.model.params.linear_end, 
            num_diffusion_timesteps=self.num_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
    
    def get_beta_schedule(self, beta_start, beta_end, num_diffusion_timesteps):
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    def image_editing_sample(self):
        print("Loading Stable Diffusion model") 
        device = self.device 

        ## Assuming Stable Diffusion 2.0
        config_path = "stable_diffusion/768-v-ema.yaml"
        ckpt_path = "stable_diffusion/768-v-ema.ckpt"

        config = OmegaConf.load(config_path) 
        model = load_model_from_config(config, ckpt_path, device)
        print("Model loaded")

        image_folder = os.path.join(self.args.image_folder, self.args.name, "img")
        mask_folder = os.path.join(self.args.image_folder, self.args.name, "mask")

        img_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])
        
        assert len(img_files) == len(mask_files)

        n = self.config.sampling.batch_size
        model.eval()
        model.float()
        print("Start sampling")

        for img_file, mask_file in zip(img_files, mask_files):
            img_path = os.path.join(image_folder, img_file)
            mask_path = os.path.join(mask_folder, mask_file)

            print(f"Processing {img_file} and {mask_file}")

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            def pad_to_multiple_of_64(image):
                w, h = image.size 
                new_w = (w // 64) * 64
                if w % 64 != 0:
                    new_w += 64
                new_h = (h // 64) * 64 
                if h % 64 != 0:
                    new_h += 64 
                pad_w = new_w - w 
                pad_h = new_h - h 
                padding = (0, 0, pad_w, pad_h) 
                return transforms.functional.pad(image, padding, fill = 0)
            
            img = pad_to_multiple_of_64(img)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            x0 = preprocess(img).unsqueeze(0).to(device).float()

            mask = pad_to_multiple_of_64(mask)
            mask = transforms.ToTensor()(mask).unsqueeze(0).to(device).float()

            # Encode image to latent space 
            with torch.no_grad():
                z = model.get_first_stage_encoding(model.encode_first_stage(x0))
            
            # Prepare mask in latent space 
            downsampling_factor = 8 
            mask_latent = F.interpolate(mask.float(), size=(z.shape[2], z.shape[3]), mode='nearest')

            total_noise_levels = self.num_timesteps
            start_step = int(self.num_timesteps * self.args.t / 1000)

            a = (1 - self.betas).cumprod(dim=0)
            e = torch.randn_like(z).float()
            z_noisy = z * a[start_step].sqrt() + e * (1.0 - a[start_step]).sqrt()

            with tqdm(total=start_step, desc="Denoising") as progress_bar:
                for i in reversed(range(start_step)):
                    t = torch.full((n,), i, device=device, dtype=torch.float32) 

                    text_prompt = ["A park with dirt"] * n
                    uc = model.get_learned_conditioning(text_prompt).float()
                    cond = {"c_crossattn": [uc]}

                    # noise_pred = model.apply_model(z_noisy.float(), t.float(), cond)
                    z_noisy = model.p_sample(z_noisy, cond, t)

                    # z_noisy = z * mask_latent + z_noisy * (1 - mask_latent)
                    z_noisy = z * (1 - mask_latent) + z_noisy * mask_latent

                    # if (i % 50) == 0 or i == 0: 
                    #     x_decoded = model.decode_first_stage(z_noisy)
                    #     x_output = (x_decoded + 1.0) / 2.0 
                    #     save_image(x_output, os.path.join(self.args.exp, f'step_{i}.png'))

                    progress_bar.update(1)

            
            x_decoded = model.decode_first_stage(z_noisy)
            x_output = (x_decoded + 1.0) / 2.0 

            id = img_file.split('.')[0]


            save_image(x_output, os.path.join(self.args.exp, f'{id}.png'))

        print("Processing completed for all images")