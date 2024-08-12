# this file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
# written by Jeong et al.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def requires_grad_(args, model:torch.nn.Module, requires_grad:bool) -> None:
    if args.ft_method == "lora":
        for name, param in model.classifier.named_parameters():
            lora_weight = ('linear_a_q.weight','linear_a_v.weight','linear_b_q.weight','linear_b_v.weight')
            #Fine-tuning only LoRA weights and MLP head 
            if name.startswith('lora_vit.head') or name.endswith(lora_weight):
                param.requires_grad_(requires_grad)
            else: 
                param.requires_grad_(False)
                
    elif args.ft_method == "full-ft":
        for param in model.parameters(): 
            param.requires_grad_(requires_grad)
    else:
        raise NotImplementedError


def copy_code(outdir):
    """Copies files to the outdir to store complete script with each experiment"""
    # embed()
    code = []
    exclude = set([])
    for root, _, files in os.walk("./code", topdown=True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]

    for r, f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.mkdir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))


def normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


def check_spectral_norm(m, name='weight'):
    from torch.nn.utils.spectral_norm import SpectralNorm
    for k, hook in m._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            return True
    return False


def apply_spectral_norm(m):
    from torch.nn.utils import spectral_norm
    for layer in m.modules():
        if isinstance(layer, nn.Conv2d):
            spectral_norm(layer)
        elif isinstance(layer, nn.Linear):
            spectral_norm(layer)
        elif isinstance(layer, nn.Embedding):
            spectral_norm(layer)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

