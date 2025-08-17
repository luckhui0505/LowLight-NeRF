# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import logging
import os
import shutil
from typing import *

import gin
import torch
from pytorch_lightning import Trainer   # PyTorch Lightning 训练器c
from pytorch_lightning import loggers as pl_loggers    # 日志记录器
from pytorch_lightning import seed_everything    # 随机种子设置
from pytorch_lightning.callbacks import (
    LearningRateMonitor,      # 学习率监控回调
    ModelCheckpoint,          # 模型检查点回调c
    TQDMProgressBar,         # 模型检查点回调
)
from pytorch_lightning.plugins import DDPPlugin    # 分布式数据并行插件
from utils.select_option import select_callback, select_dataset, select_model     # 自定义选择函数
 
# 字符串转布尔值的辅助函数
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")

# 使用gin配置系统装饰的主运行函数
@gin.configurable()
def run(
    # 基本配置参数
    ginc: str,      # gin配置文件路径
    ginb: str,       # gin绑定参数
    resume_training: bool,    # 是否恢复训练
    ckpt_path: Optional[str],     # 检查点路径(可选)
    # 检查点路径(可选)
    scene_name: Optional[str] = None,    #场景名称(可选)
    datadir: Optional[str] = None,      # 数据目录(可选)
    logbase: Optional[str] = None,       # 日志基础目录(可选)
    # 模型名称(可选)
    model_name: Optional[str] = None,     # 模型名称(可选)
    dataset_name: Optional[str] = None,     # 数据集名称(可选)
    # 其他选项
    postfix: Optional[str] = None,      # 实验名称后缀(可选)
    entity: Optional[str] = None,        # 实体名称(如W&B)(可选)
    # Hyper Parameter in Aleth-NeRF  超参数
    #con: float = 12,    # set here to 2 for over-exposure conditions 对比度控制参数(过曝光条件设为2)
    #eta: float = 0.45, #隐藏控制度
    # Optimization
    max_steps: int = -1,
    max_epochs: int = -1,
    precision: int = 32,
    # Logging
    log_every_n_steps: int = 1000,  # 日志记录频率
    progressbar_refresh_rate: int = 5,  # 进度条刷新率
    # Run Mode  运行模式控制
    run_train: bool = True,  # 是否运行训练
    run_eval: bool = True,    # 是否运行评估
    run_render: bool = False,   # 是否运行渲染
    # 设备设置
    num_devices: Optional[int] = None,   # 使用设备数量
    num_sanity_val_steps: int = 0,    #验证前 sanity check 步数
    # 其他
    seed: int = 777,     # 随机种子
    debug: bool = False,   # 是否调试模式
    save_last: bool = True,   # 是否保存最后检查点c
    grad_max_norm=0.0,   # 梯度裁剪最大值
    grad_clip_algorithm="norm",   # 梯度裁剪算法
):
    # 打印当前场景名称
    print('the scene name is:', scene_name)
    # 设置lightning日志级别为ERROR
    logging.getLogger("lightning").setLevel(logging.ERROR)
    # 去除数据目录末尾的斜杠
    datadir = datadir.rstrip("/")
    try:
        eta = gin.query_parameter("LitAleth_NeRF.eta")
        con = gin.query_parameter("LitAleth_NeRF.con")
    except ValueError:
        eta, con = 0.45, 12  # 默认值
    
    
    print(f"[Config Check] eta={eta}, con={con}")
    # 构建实验名称: 模型_数据集_场景_eta值_con值
    exp_name = (model_name + "_" + dataset_name + "_" + scene_name + "_" +  \
        'eta' + str(eta) + 'con' + str(con))
    # 如果有后缀则添加到实验名称
    if postfix is not None:
        exp_name += "_" + postfix
    if debug:
        exp_name += "_debug"
    # 如果未指定设备数量，使用所有可用GPU
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    # 如果是plenoxel模型，强制使用单设备
    if model_name in ["plenoxel"]:
        num_devices = 1
    # 如果未指定日志基础目录，使用默认值
    if logbase is None:
        logbase = "/logs"
    # 创建日志基础目录
    os.makedirs(logbase, exist_ok=True)
    # 创建实验日志目录
    logdir = os.path.join(logbase, exp_name)
    os.makedirs(logdir, exist_ok=True)
    # 创建实验子目录
    os.makedirs(os.path.join(logdir, exp_name), exist_ok=True)
    # 打印日志目录路径
    print('the log dir is:', logdir)
    # 初始化TensorBoard日志记录器
    logger = pl_loggers.TensorBoardLogger(
        save_dir=logdir,
        name=exp_name,
    )
    # Logging all parameters
    # 如果是训练模式，记录所有配置参数
    if run_train:
        # 配置文件保存路径
        txt_path = os.path.join(logdir, "config.gin")
        # 写入配置文件内容
        with open(txt_path, "w") as fp_txt:
            # 写入每个配置文件的路径和内容
            for config_path in ginc:
                fp_txt.write(f"Config from {config_path}\n\n")
                with open(config_path, "r") as fp_config:
                    readlines = fp_config.readlines()
                for line in readlines:
                    fp_txt.write(line)
                fp_txt.write("\n")

            fp_txt.write("\n### Binded options\n")
            for line in ginb:
                fp_txt.write(line + "\n")

    seed_everything(seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
     # 初始化模型检查点回调
    model_checkpoint = ModelCheckpoint(
        monitor="val/psnr",   # 监控验证PSNR
        dirpath=logdir,
        filename="best",
        save_top_k=1,  # 保存最好的1个模型
        mode="max",   # 最大化监控指标
        save_last=save_last,   # 是否保存最后检查点
    )
     # 初始化进度条回调
    tqdm_progrss = TQDMProgressBar(refresh_rate=progressbar_refresh_rate)

    callbacks = []
    # 如果不是plenoxel模型，添加学习率监控
    if not model_name in ["plenoxel"]:
        callbacks.append(lr_monitor)
    callbacks += [model_checkpoint, tqdm_progrss]
    callbacks += select_callback(model_name)
    # 如果是多设备训练，初始化DDP插件
    ddp_plugin = DDPPlugin(find_unused_parameters=False) if num_devices > 1 else None
    # 初始化PyTorch Lightning训练器
    trainer = Trainer(
        logger=logger if run_train else None,  # 训练时使用日志记录器
        log_every_n_steps=log_every_n_steps,   # 日志记录频率
        devices=num_devices,     # 使用设备数量
        max_epochs=max_epochs,  #-1  最大训练周期数
        max_steps=max_steps,    # 最大训练步数
        accelerator="gpu",   # 使用GPU加速
        replace_sampler_ddp=False,    # 不替换DDP采样器
        strategy=ddp_plugin,    # 分布式策略
        check_val_every_n_epoch=1,    # 每1个epoch验证一次
        precision=precision,    # 训练精度
        num_sanity_val_steps=num_sanity_val_steps,   # 验证前sanity check步
        callbacks=callbacks,     # 回调函数列表
        gradient_clip_algorithm=grad_clip_algorithm,    # 梯度裁剪算法
        gradient_clip_val=grad_max_norm,    # 梯度裁剪值
    )

    if resume_training:
        if ckpt_path is None:
            ckpt_path = f"{logdir}/last.ckpt"

    data_module = select_dataset(
        dataset_name=dataset_name,
        scene_name=scene_name,
        datadir=datadir,
    )

    model = select_model(model_name=model_name) # model define
    model.logdir = logdir
    if run_train:   # Training
        best_ckpt = os.path.join(logdir, "best.ckpt")
        if os.path.exists(best_ckpt):
            os.remove(best_ckpt)
        version0 = os.path.join(logdir, exp_name, "version_0")
        if os.path.exists(version0):
            shutil.rmtree(version0, True)

        trainer.fit(model, data_module, ckpt_path=ckpt_path)

    if run_eval:    # Evaluation
        ckpt_path = (
            f"{logdir}/last.ckpt"
            if dataset_name != 'single_image_non_ref'
            else f"{logdir}/last.ckpt"
        )
        print('the checkpoint path is:', ckpt_path)
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    if run_render:  # Rendering
        print('rendering')
        ckpt_path = (
            f"{logdir}/last.ckpt"
            #if model_name != "mipnerf360" or dataset_name != 'single_image_non_ref'
            if dataset_name != 'single_image_non_ref'
            else f"{logdir}/last.ckpt"
        )
        print('the checkpoint path is:', ckpt_path)
        trainer.predict(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument(
        "--resume_training",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="gin bindings",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to checkpoints"
    )

    parser.add_argument("--seed", type=int, default=220901, help="seed to use")
    parser.add_argument("--logbase", type=str, default="./logs", help="seed to use")
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)

    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")

    gin.parse_config_files_and_bindings(args.ginc, ginbs)
    run(
        logbase=args.logbase,
        ginc=args.ginc,
        ginb=ginbs,
        resume_training=args.resume_training,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
    )
