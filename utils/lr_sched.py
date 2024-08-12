# this file is based on code publicly available at
# https://github.com/facebookresearch/mae
# written by Kaiming He et al.

import math

def adjust_learning_rate_weight_decay(optimizer, global_iter, global_iter_eplike, wd_schedule, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if global_iter_eplike < args.warmup_epochs:
        lr = args.lr * global_iter_eplike / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * (global_iter_eplike - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        #Layer-decay
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
        
        #Weight-decay
        if "weight_decay" in param_group and param_group["weight_decay"] != 0.:
            param_group["weight_decay"] = wd_schedule[global_iter]

    return lr