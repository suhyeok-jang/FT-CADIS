# Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness

This repository contains code for the paper "Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness" by Suhyeok Jang, Seojin Kim, Jinwoo Shin and Jongheon Jeong.

<b>TL;DR:</b>  *Enhancing the certified robustness of the smoothed classifier by fine-tuning the off-the-shelf model on selectively chosen denoised images.* 

![main_figure (1)](https://github.com/user-attachments/assets/885eda34-ad32-40d4-a251-ac3d8bb4ff62)

## Environmental setup
Clone this repo and install the dependencies with provided conda yaml file:
We used Python 3.8.19, PyTorch 2.2.0, Torchvision 0.17.0, and Timm 0.9.16 as the default settings.
```
git clone https://github.com/suhyeok24/ft-cadis.git
conda env create --file conda-environment.yaml
```