# Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness

This repository contains code for the paper "Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness". 

<b>TL;DR:</b>  *Enhancing the certified robustness of the smoothed classifier by fine-tuning the off-the-shelf model on selectively chosen denoised images.* 

![main_figure (1)](https://github.com/user-attachments/assets/885eda34-ad32-40d4-a251-ac3d8bb4ff62)

## Setup
Set up a new conda virtual environment for <b>ft-cadis</b> on Python 3.8.19. The default settings include [PyTorch](https://pytorch.org/) 2.2.0, [Torchvision](https://pytorch.org/vision/stable/index.html) 0.17.0, and [Timm](https://github.com/huggingface/pytorch-image-models) 0.9.16.
```
conda create -n ft-cadis python=3.8.19 -y
conda activate ft-cadis
bash setup_environment.sh
```
Additionally, we utilize the same denoiser as [Carlini et al. (2023)](https://arxiv.org/abs/2206.10550). Please make sure to download the appropriate model checkpoints for each dataset from the respective repo:
[CIFAR-10](https://github.com/openai/improved-diffusion): Unconditional CIFAR-10 with `L_hybrid` objective and cosine noise schedule
[ImageNet](https://github.com/openai/guided-diffusion): 256x256 diffusion (not class conditional)

## Training
We offer an example command line input to run `train.py` on CIFAR-10 and ImageNet.
```
# CIFAR-10 (Multi-GPU)
bash train_cifar10.sh --ngpus [NUM OF GPUS] --noise 1.00 --blr 1e-4 --batch 32 --accum_iter 4 --lbd 4.0

# ImageNet (Multi-GPU)
bash train_imagenet.sh --ngpus [NUM OF GPUS] --noise 1.00 --blr 4e-4 --batch 16 --accum_iter 4 --lbd 2.0
``` 
- The default base learning rate `blr` and coefficient for masked adversarial loss `lbd` are provided in [our paper](https://openreview.net/pdf?id=99GovbuMcP).
- Here the efffective batch size is 128:
    - It is calucated as `ngpus` x `batch` per gpu x `accum_iter` // `num_noises`
    - Increase `accum_iter` to maintain the effective batch size if VRAM or the number of GPUs is limited.
- To resume fine-tuning from a specific checkpoint, use the `resume` and `load_from` arguments.
    ```
    bash train_cifar10.sh --ngpus [NUM OF GPUS] --noise 1.00 --blr 1e-4 --batch 32 --accum_iter 4 --lbd 4.0 \
    --resume --load_from [CHECKPOINT LOCATION]
    ```

## Certification
We provide a sample command to perform certification on CIFAR-10 and ImageNet based on [Cohen et al. (2019)](https://github.com/locuslab/smoothing?tab=readme-ov-file)
```
# CIFAR-10 (Single-GPU)
python certify.py --seed 0 --dataset cifar10 --sigma 0.50 --skip 1 --N0 100 --N 100000 --batch_size 400 --finetuned_path [CHECKPOINT LOCATION] --outfile [OUTPUT LOCATION]

# ImageNet (Multi-GPU)
python certify.py -seed 0 --dataset imagenet --sigma 0.50 --skip 1 --N0 100 --N 10000 --batch_size 32 --finetuned_path [CHECKPOINT LOCATION] --outfile [OUTPUT LOCATION]
```

## Others
The `analyze.py` includes various helpful classes and functions for analyzing and visualizing certification results, outputting it in LaTeX, table, or graph format.

## Acknowledgments
This repository is built on top of [Diffusion Denoised](https://github.com/ethz-spylab/diffusion_denoised_smoothing), [Multi-scale Denoised](https://github.com/jh-jeong/smoothing-multiscale) and [CAT-RS](https://github.com/alinlab/smoothing-catrs).