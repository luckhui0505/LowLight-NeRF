# LowLight-NeRF


## 1.Install environment
```
git clone https://github.com/luckhui0505/LowLight-NeRF.git
cd LowLight-NeRF
conda create -n aleth_nerf -c anaconda python=3.8
conda activate aleth_nerf
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
## 2. Abstract

Neural Radiance Fields (NeRF) can generate high-quality novel views synthesis using multi-view images, but its performance is limited in low-light scenes. To address the issues of insufffcient reconstruction accuracy and inadequate detail recovery in low-light environments, we propose a new NeRF framework for low-light scenes, called LowLight-NeRF. First, LowLight-NeRF introduces Ambient Light Simulation (ALS), which simulates natural ambient lighting information to restore missing lighting data in images, thereby effectively enhancing the natural appearance and 3D reconstruction accuracy of images in low-light scenes. Second, LowLight-NeRF introduces the Lowlight Area Enhancement (LAE) method, which adaptively selects low-light areas and enhances their brightness, thereby more accurately restoring detail information in low-light scenes. Finally, LowLight-NeRF employs a Custom Contrast-aware Loss (CA Loss) function to enhance the network’s ability to perceive image details in low-brightness regions, signiffcantly improving 3D reconstruction performance. Through qualitative and quantitative analysis, the experimental results show that our method signiffcantly improves the PSNR, SSIM, and LPIPS metrics, with the most notable improvement in PSNR, which is 3.93% higher than the SOTA average. The source code is available at: https://github.com/luckhui0505/LowLight-NeRF.

## 2. Comparison of Experimental Results
![image](https://github.com/luckhui0505/LowLight-NeRF/figure1.jpg) 

## 3. Training Aleth-NeRF


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py --ginc configs/LOM/aleth_nerf/aleth_nerf_buu.gin --logbase ./logs 
```

You can also direct use following command to run all 5 scenes scenes together:

```
bash run/run_LOM_aleth.sh
```




## 4. Setting parameters
Changing the data path and log path in the configs/demo_blurfactory.txt

## Some Notes
### GPU Memory
We train our model on a RTX3090 GPU with 24GB GPU memory. If you have less memory, setting N_rand to a smaller value, or use multiple GPUs.
## Acknowledge
Code is based on Aleth-Nerf, much thanks to their excellent codebase! 











