# Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness

This repository contains code for the paper "Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness" 

<b>TL;DR:</b>  *Enhancing the certified robustness of the smoothed classifier by fine-tuning the off-the-shelf model on selectively chosen denoised images.* 

![main_figure (1)](https://github.com/user-attachments/assets/885eda34-ad32-40d4-a251-ac3d8bb4ff62)

## Setup
Set up a new conda virtual environment for <b>ft-cadis</b> on Python 3.8.19. The default settings include [PyTorch](https://pytorch.org/) 2.2.0, [Torchvision](https://pytorch.org/vision/stable/index.html) 0.17.0, and [Timm](https://github.com/huggingface/pytorch-image-models) 0.9.16.
```
conda create -n ft-cadis python=3.8.19 -y
conda activate ft-cadis
bash setup_environment.sh
```

## Training
We offer an example command line input to run `train.py` on CIFAR-10 and ImageNet.
```
# CIFAR-10 (Multi-GPU)
bash train_cifar10.sh --ngpus [NUM of GPUS] --noise 1.00 --blr 1e-4 --batch 32 --accum_iter 4 --lbd 4.0

#ImageNet (Multi-GPU)
bash train_imagenet.sh --ngpus [NUM of GPUS] --noise 1.00 --blr 4e-4 --batch 16 --accum_iter 4 --lbd 2.0
``` 
- The default base learning rate `blr` and coefficient for masked adversarial loss `lbd` are provided in [our paper](https://openreview.net/pdf?id=99GovbuMcP).
- Here the efffective batch size is 128:
    - It is calucated as ngpus x `batch` per gpu x `accum_iter` // `num_noises`
    - Increase `accum_iter` to maintain the effective batch size if VRAM or the number of GPUs is limited.

## Certification

## Acknowledgments