# this file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
# written by Jeong et al.
# https://github.com/ethz-spylab/diffusion_denoised_smoothing
# written by Carlini et al.

from typing import Optional
import torch
import torch.nn as nn
import timm
from transformers import AutoModelForImageClassification

from third_party.lora_vit.lora import LoRA_ViT_timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCHITECTURES = ["cifar10_vit_base", "imagenet_vit_base"]


def get_architecture(
    arch: str, ft_method: Optional[str] = None, drop_path_rate: Optional[float] = None
) -> torch.nn.Module:
    """Return a neural network"""
    if arch == "cifar10_vit_base":
        model = ViT_base_patch16_224(ft_method, drop_path_rate)

    elif arch == "imagenet_vit_base":
        model = ViT_base_patch16_384(ft_method, drop_path_rate)

    else:
        raise KeyError(f"Unknown architecture '{arch}'")

    return model.to(device)


class ViT_base_patch16_224(nn.Module):
    def __init__(self, ft_method=None, drop_path_rate=None, depth=12):
        super().__init__()

        classifier = AutoModelForImageClassification.from_pretrained(
            "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        )

        if ft_method is not None:
            if ft_method == "full-ft":
                for param in classifier.parameters():
                    param.requires_grad = True
            else:
                raise NotImplementedError

        if drop_path_rate is not None:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay rule
            # Apply dropout rates for each block layer
            for i, layer in enumerate(classifier.vit.encoder.layer):
                layer.attention.attention.dropout.p = dpr[i]

        classifier.cuda()
        self.classifier = classifier

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, (224, 224), mode="bicubic", antialias=True)

        out = self.classifier(x)
        out = out.logits

        return out


class ViT_base_patch16_384(nn.Module):
    def __init__(self, ft_method=None, drop_path_rate=None):
        super().__init__()

        # CLIP pretrained image tower and related fine-tuned weights
        classifier = timm.models.create_model(
            "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
            pretrained=True,
            drop_path_rate=drop_path_rate,
        )

        if ft_method is not None:
            if ft_method == "full-ft":
                for param in classifier.parameters():
                    param.requires_grad = True
            elif ft_method == "lora":
                classifier = LoRA_ViT_timm(vit_model=classifier, r=4, alpha=4, num_classes=1000)
            else:
                raise NotImplementedError

        classifier.cuda()
        self.classifier = classifier

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, (384, 384), mode="bicubic", antialias=True)

        out = self.classifier(x)

        return out
