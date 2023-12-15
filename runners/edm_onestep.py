import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion import gaussian_diffusion

import math
import pickle
import dnnlib

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
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class EDM_onestep(torch.nn.Module):
    def __init__(self, sigma, device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        print("Loading model")
        model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
        with dnnlib.util.open_url(f'{model_root}/edm-cifar10-32x32-uncond-vp.pkl') as f:
            model = pickle.load(f)['ema'].to(self.device).eval()


        # defaults = model_and_diffusion_defaults(self.args.t_total)
        # model, diffusion = create_model_and_diffusion(**defaults)
        # model.load_state_dict(
        #     dist_util.load_state_dict("pretrained/cifar10_uncond_50M_500K.pt", map_location="cpu")
        # )
        # model.to(self.device)
        # model.eval()

        # self.model = model
        # self.diffusion = diffusion

        self.model = model
        self.scale = 1

        self.sigma = sigma
        sigma = sigma*2

        num_steps = 18
        sigma_max = 80
        sigma_min = 0.002
        rho = 7
        sigma_min = max(sigma_min, self.model.sigma_min)
        sigma_max = min(sigma_max, self.model.sigma_max)
        print("sigma: ",sigma)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        self.t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.t_steps = torch.cat([self.model.round_sigma(self.t_steps), torch.zeros_like(self.t_steps[:1])]) # t_N = 0
        for i in range(len(self.t_steps)-1):
            if self.t_steps[i]>=sigma and self.t_steps[i+1]<sigma:
                if self.t_steps[i]-sigma > sigma-self.t_steps[i+1]:
                    self.t = i+1
                    break
                else:
                    self.t = i
                    break
            self.t = len(self.t_steps)-1

    def image_editing_sample(self, img=None, bs_id=0, tag=None, sigma=0.0):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            # out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            x0 = img

            x0 = self.scale*(img)
            t = self.t

            x_next = x0
            for i in range(len(self.t_steps)-self.t-1):
                # Increase noise temporarily.
                # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                # t_hat = net.round_sigma(t_cur + gamma * t_cur)
                # x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
                x_hat = x_next
                t_hat = self.t_steps[t+i]
                t_next = self.t_steps[t+i+1]

                # Euler step.
                denoised = self.model(x_hat, t_hat).to(torch.float64)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # break # one step
            
                 # Apply 2nd order correction.
                if i < len(self.t_steps)-self.t-2:
                    denoised = self.model(x_next, t_next).to(torch.float64)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                    print("Apply 2nd order correction.")
                break

            x0 = x_next

            
            # save purified images
            x_ = ((x0+1) * 127.5).clamp(0, 255).to(torch.uint8)
            x_ = x_.permute(0, 2, 3, 1)
            x_ = x_.contiguous()
            all_images = []
            all_images.extend([x_.cpu().numpy()])
            arr = np.concatenate(all_images, axis=0)
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join("/home/yiquan/DensePure/purified_images", f"testing_samples_{shape_str}.npz")
            np.savez(out_path, arr)
            print("images saved!")
            
            return x0