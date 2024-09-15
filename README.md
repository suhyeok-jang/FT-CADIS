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

## Certification

## Acknowledgments