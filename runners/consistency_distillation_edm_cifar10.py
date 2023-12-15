import os
import random

import numpy as np

import torch
import torchvision.utils as tvu
import argparse
from PIL import Image
#from cm import dist_util

from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.karras_diffusion import karras_sample

def create_argparser_cm():
    defaults = dict(
        training_mode="consistency_distillation",
        generator="determ",
        clip_denoised=True,
        num_samples=500,
        batch_size=32,
        sampler="multistep_purification",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=60,
        model_path="/home/yiquan/consistency_models/results/train_cifar10/openai-2023-07-12-00-04-10-431848/ema_0.9999_090000.pt",
        # model_path = "/home/yiquan/consistency_models/results/train_cifar10/openai-2023-12-05-12-29-50-791308/target_model084000.pt",
        seed=42,
        ts=[37,59],
        noise_sigma=0.5,
        data_dir="",
        image_size=32,
        attention_resolutions="16",
        class_cond=False,
        dropout=0.1,
        num_channels=128,
        num_res_blocks=3,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
        weight_schedule="karras",
        sigma_min=0.002,
        sigma_max=80.0,
        channel_mult="",
        learn_sigma=False,
        use_new_attention_order=False,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        use_checkpoint=False,
    )
    # defaults.update(model_and_diffusion_defaults())
    
    return defaults
    
class ConsistencyDiffusion(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.args = create_argparser_cm()

        model, diffusion = create_model_and_diffusion(
            image_size=self.args['image_size'],
            class_cond=self.args['class_cond'],
            learn_sigma=self.args['learn_sigma'],
            num_channels=self.args['num_channels'],
            num_res_blocks=self.args['num_res_blocks'],
            channel_mult=self.args['channel_mult'],
            num_heads=self.args['num_heads'],
            num_head_channels=self.args['num_head_channels'],
            num_heads_upsample=self.args['num_heads_upsample'],
            attention_resolutions=self.args['attention_resolutions'],
            dropout=self.args['dropout'],
            use_checkpoint=self.args['use_checkpoint'],
            use_scale_shift_norm=self.args['use_scale_shift_norm'],
            resblock_updown=self.args['resblock_updown'],
            use_fp16=self.args['use_fp16'],
            use_new_attention_order=self.args['use_new_attention_order'],
            weight_schedule=self.args['weight_schedule'],
            sigma_min=self.args['sigma_min'],
            sigma_max=self.args['sigma_max'],
            distillation=True,
        )
        print("Loading model")
        model.load_state_dict(torch.load(self.args['model_path']))
        model.to(device)
        if self.args['use_fp16']:
            model.convert_to_fp16()
        model.eval()
        
        self.model=model
        self.diffusion=diffusion
        
        
        
    def image_editing_sample(self, img=None, bs_id=0, tag=None, sigma=0.0):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        image_size = img.shape[-1]
        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            # out_dir = os.path.join(self.args['log_dir'], 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            corrupted_images = img
            
            sample = ((corrupted_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            
            for i in range(sample.shape[0]):
                img = Image.fromarray(sample[i].cpu().numpy(), 'RGB')
                
                img.save(f'/home/yiquan/DensePure/purified_images/corrupted/image_{i}.png')
            
            
            all_images = []
            x0 = karras_sample(
                self.diffusion,
                self.model,
                (batch_size, 3, image_size, image_size),
                steps=self.args['steps'],
                model_kwargs={},
                device=self.device,
                clip_denoised=self.args['clip_denoised'],
                sampler=self.args['sampler'],
                sigma_min=self.args['sigma_min'],
                sigma_max=self.args['sigma_max'],
                s_churn=self.args['s_churn'],
                s_tmin=self.args['s_tmin'],
                s_tmax=self.args['s_tmax'],
                s_noise=self.args['s_noise'],
                generator=None,
                ts=self.args['ts'],
                purification=True,
                corrupted_images=corrupted_images,
                noise_sigma=sigma*2
            )
            
            
            sample = ((x0 + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            
            for i in range(sample.shape[0]):
                img = Image.fromarray(sample[i].cpu().numpy(), 'RGB')
                
                img.save(f'/home/yiquan/DensePure/purified_images/failcase/image_{i}.png')
            
        return x0
        
    
    