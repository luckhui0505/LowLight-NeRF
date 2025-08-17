# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from random import random, sample
from turtle import forward
from typing import *

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt


import src.model.aleth_nerf.helper as helper
import utils.store_image as store_image
from src.model.interface import LitModel    # Notice Here

# Positional encoding (section 5.1) 位置编码
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)

def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

@gin.configurable()
class NeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,  #位置编码的最低和最高频率
        max_deg_point,    
        deg_view,   #视角方向编码的度数
        netdepth: int = 8,  #网络深度（层数）
        netwidth: int = 256,      #网络宽度（每层的神经元数）
        netdepth_condition: int = 1,   #条件分支的深度
        netdepth_dark: int = 3,      #暗度分支的深度c
        netwidth_condition: int = 128,    #条件分支的宽度
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        num_dark_channels: int = 1,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        #位置编码处理
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        

        pts_linear = [init_layer]
        
        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:   
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)
        
        
        self.pts_linears = nn.ModuleList(pts_linear)    

        dark_linear = [nn.Linear(netwidth, netwidth_condition)]
        for idx in range(netdepth_dark - 1):
            dark = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(dark.weight)
            dark_linear.append(dark)

        self.dark_linear = nn.Sequential(*dark_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):   # 0
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)
        
        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)
        self.dark_layer = nn.Linear(netwidth_condition, num_dark_channels)
        self.dark_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3)
        # 新增环境光强度预测层
        self.ambient_layer = nn.Linear(netwidth_condition, 1)
        self.ambient_activation = lambda x: torch.sigmoid(x) * 0.06     # 使用 Sigmoid 后乘以 0.1 限制在 [0, 0.1] 范围
        init.xavier_uniform_(self.ambient_layer.weight)

        
        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)
        init.xavier_uniform_(self.dark_layer.weight)


    def forward(self, x, condition):
        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)
        inputs = x

        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1) 
        

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        
        raw_darkness = self.dark_linear(x)
        raw_darkness = self.dark_layer(raw_darkness)
        raw_darkness = raw_darkness.reshape(-1, num_samples, self.num_density_channels).permute(1,2,0) # (8192, 65, 1)
        raw_darkness = self.dark_conv(raw_darkness).permute(2,0,1)

        bottleneck = self.bottleneck_layer(x)
        
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)
        # 预测环境光强度（新增的核心代码）
        ambient_strength = self.ambient_activation(self.ambient_layer(x))  # ← 您问的这行
        ambient_strength = ambient_strength.reshape(-1, num_samples, 1)  # 新增reshape

        # 生成原始RGB输出
        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)
    
        # 将环境光加到 RGB 输出中
        ambient_rgb = raw_rgb * ambient_strength  # 环境光的影响
        output_rgb  =raw_rgb + ambient_rgb  # 将环境光加到原始 RGB 上
        
        return output_rgb, raw_density, raw_darkness, ambient_strength

@gin.configurable()
class Aleth_NeRF(nn.Module):
    def __init__(
        self,
        K_g: float = 0.5,   # Initial Global Density Value   # 全局暗度初始值
        num_levels: int = 2,     # 网络层级数（粗+细）
        min_deg_point: int = 0, # 位置编码最小频率
        max_deg_point: int = 10, # 位置编码最大频率
        deg_view: int = 4,     # 视角方向编码度数
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,   # 是否使用视角方向
        noise_std: float = 0.0,  # 噪声标准差
        lindisp: bool = False,   # 是否在视差空间线性采样
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(Aleth_NeRF, self).__init__()
        
        self.rgb_activation = nn.Sigmoid()
        self.sigma_activation = nn.ReLU()
        self.dark_activation = nn.Sigmoid()
        #self.dark_activation = nn.ReLU()
        # 全局暗度参数 - Aleth-NeRF的创新点1
        self.K_g = K_g  # global conceiling field initial value  # 全局暗度初始值
        self.coarse_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)   
        self.fine_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        # global conceiling field # 可学习的全局暗度参数 - 创新点2
        self.coarse_dark = nn.Parameter(K_g * torch.ones(num_coarse_samples + 1), requires_grad=True) # 65
        self.fine_dark = nn.Parameter(K_g * torch.ones(num_fine_samples + num_coarse_samples + 1), requires_grad=True)  # 193
        


    def forward(self, rays, randomized, white_bkgd, near, far, mode): # white_bkgd: False; near: None; far: None
        
        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                # sampling point, sampling point's cordinate
                t_vals, samples = helper.sample_along_rays(         # coarse ray sampling
                    rays_o=rays["rays_o"],  # ray's initial points [1024, 3]
                    rays_d=rays["rays_d"],  # ray's direction   [1024, 3]
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                mlp = self.coarse_mlp
            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples = helper.sample_pdf(               # fine ray re-sampling
                    bins=t_mids,
                    weights=weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                )
                mlp = self.fine_mlp

            # samples position encoding
            samples_enc = helper.pos_enc(
                samples,    # (1024, N, 3) 的 tensor
                self.min_deg_point,
                self.max_deg_point,
            )  

            viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view) 
             # 可学习的全局暗度参数 - 创新点2
            raw_rgb, raw_sigma, raw_darkness, ambient_strength = mlp(samples_enc, viewdirs_enc)
            
            if self.noise_std > 0 and randomized:   # False
                raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std
            
            rgb = self.rgb_activation(raw_rgb) + 1e-4  # nn.Sigmoid() 
            sigma = self.sigma_activation(raw_sigma)    # nn.ReLU() 

            # Local Conceling Field 
            darkness = self.dark_activation(raw_darkness)   # nn.Sigmoid()
            
            # Volumn Rendering in Training Stage, with concealing fileds
            if mode != 'test':
                comp_rgb_dark, depth, weights, comp_rgb = helper.volumetric_rendering(
                    rgb,    # [B, 65/129, 3]
                    sigma,  # [B, 65/129, 1]
                    darkness,   # [B, 65/129, 1]
                    t_vals, 
                    rays["rays_d"],
                    mode,   
                    i_level,    
                    self.coarse_dark,
                    self.fine_dark,
                    white_bkgd=white_bkgd,
                )

                ret.append((comp_rgb_dark, darkness, sigma, depth, weights, comp_rgb))

            # Volumn Rendering in Testing Stage, without concealing fileds
            else:   
                comp_rgb, depth, weights = helper.volumetric_rendering(
                    rgb,    # [B, 65/129, 3]
                    sigma,  # [B, 65/129, 1]
                    darkness,   # [B, 65/129, 1]
                    t_vals, 
                    rays["rays_d"],
                    mode,   # train & test
                    i_level,    
                    self.coarse_dark,
                    self.fine_dark,
                    white_bkgd=white_bkgd,
                )

                ret.append((comp_rgb, depth, sigma, darkness))

        return ret


@gin.configurable()
class LitAleth_NeRF(LitModel):
    def __init__(
        self,
        eta: float = 0.4,         # Enhance Lightness Degree (0.35~0.55 all OK)
        con: float = 10,         # Enhance Contrast Degree
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitAleth_NeRF, self).__init__()
        self.model = Aleth_NeRF() # put Aleth-NERF on the optimizer
        self.eta = eta
        self.con = con
        
        

    def setup(self, stage: Optional[str] = None) -> None:
        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.white_bkgd = self.trainer.datamodule.white_bkgd
    
    # For training
    def training_step(self, batch, batch_idx):
        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far, mode = 'train'
        )
        # 获取新增的环境光强度
        ambient_coarse = rendered_results[0][-1]
        ambient_fine = rendered_results[1][-1]
        
        rgb_coarse_dark, rgb_fine_dark = rendered_results[0][0], rendered_results[1][0]     # low light RGB
        l_conceil_coarse, l_conceil_fine = rendered_results[0][1], rendered_results[1][1]   # local conceiling
        l_density_coarse, l_density_fine = rendered_results[0][2], rendered_results[1][2]   # density
        rgb_coarse_light, rgb_fine_light = rendered_results[0][-1], rendered_results[1][-1] # normal light RGB
        
        target = batch["target"]
        # target_mean = torch.mean(target, dim=-1)    # RGB mean
        
        mean_rgb_coarse = torch.mean(rgb_coarse_light, dim=0)
        mean_rgb_fine = torch.mean(rgb_fine_light, dim=0)
        
        # NeRF Loss
        loss0 = helper.img2mse_tone(rgb_coarse_dark, target)
        loss1 = helper.img2mse_tone(rgb_fine_dark, target)
        # Local Conceiling Control Loss
        loss_control = helper.Exp_loss_global(mean_val=self.eta)(rgb_coarse_light) + helper.Exp_loss_global(mean_val=self.eta)(rgb_fine_light)   # twilight

        target = target.type(torch.FloatTensor).to(rgb_fine_dark.device)
        loss_structure = helper.Structure_Loss(contrast=self.eta * self.con)(target, rgb_coarse_light) + helper.Structure_Loss(contrast=self.eta * self.con)(target, rgb_fine_light)
        loss_cc = helper.colour(mean_rgb_coarse) + helper.colour(mean_rgb_fine)

         # 可以添加对环境光强度的正则化损失
        loss_ambient = torch.mean(ambient_coarse) + torch.mean(ambient_fine)
 
        
        loss = loss1 + loss0  + 1e-3*loss_control + 1e-3*loss_structure + 1e-8*loss_cc + 1e-4*loss_ambient    # dark
        
         
        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)
        
        # Logging in Tensorboard
        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss_cc", 1e-8*loss_cc, on_step=True)
        self.log("train/loss_control", 1e-3*loss_control, on_step=True)
        self.log("train/loss_structure", 1e-3*loss_structure, on_step=True)
         # 记录环境光强度
        self.log("train/ambient_coarse", torch.mean(ambient_coarse), on_step=True)
        self.log("train/ambient_fine", torch.mean(ambient_fine), on_step=True)


        return loss


    # For testing & evaluation
    def render_rays(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far, mode = 'test'
        )
        
        rgb_fine = rendered_results[1][0]
        depth_fine = rendered_results[1][1]
        depth_fine = torch.stack([depth_fine, depth_fine, depth_fine], dim=-1)
        
        #sigma = rendered_results[1][2]
        #darkness = rendered_results[1][-1]

        rgb_fine = torch.clip(rgb_fine, 0, 1)
        target = batch["target"]
        
        ret["target"] = target
        ret["rgb"] = rgb_fine
        ret["depth"] = depth_fine
        #ret["darkness"] = darkness
        #ret["sigma"] = sigma
        return ret
    
    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = gin.query_parameter("run.max_steps")

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        
        optimizer.step(closure=optimizer_closure)
    
    def validation_epoch_end(self, outputs):
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )
        
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        depths = self.alter_gather_cat(outputs, "depth", all_image_sizes)
        visual_dir = os.path.join(self.logdir, "visual")
        os.makedirs(visual_dir, exist_ok=True)
        
        
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_model")
            depth_dir = os.path.join(self.logdir, "depth")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)
            store_image.store_depth(depth_dir, depths)
            result_path = os.path.join(self.logdir, "results.json")
            self.write_stats(result_path, psnr, ssim, lpips)

        return psnr, ssim, lpips
