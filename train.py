# this file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
# written by Jeong et al.
# https://github.com/ethz-spylab/diffusion_denoised_smoothing
# written by Carlini et al.

import argparse
import time
from typing import Optional
import numpy as np
import os
import math
import sys

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from classifier import ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
from train_utils import accuracy, log, requires_grad_, init_logfile
from tensorboardX import SummaryWriter
from utils.lr_sched import adjust_learning_rate_weight_decay
import utils.lr_decay as lrd
from utils import utils
from denoiser import CIFAR10_Denoiser, ImageNet_Denoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="PyTorch FT-CADIS Training")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--id", default=None, type=int, help="experiment id")
parser.add_argument("--dataset", type=str, choices=DATASETS)
parser.add_argument("--arch", type=str, choices=ARCHITECTURES)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--warmup_epochs", type=int, default=3, help="epochs to warmup LR")
parser.add_argument("--batch", default=32, type=int, metavar="N", help="batchsize (default: 32)")
parser.add_argument(
    "--accum_iter",
    default=1,
    type=int,
    help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
)
parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument(
    "--blr",
    type=float,
    default=1e-3,
    metavar="LR",
    help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
)
parser.add_argument(
    "--min_lr",
    type=float,
    default=1e-6,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0",
)
parser.add_argument(
    "--train_noise_sd",
    default=0.25,
    type=float,
    help="standard deviation of Gaussian noise for data augmentation for train",
)
parser.add_argument(
    "--test_noise_sd",
    default=0.25,
    type=float,
    help="standard deviation of Gaussian noise for data augmentation for test",
)

#####################
# Options added by Salman et al. (2019)
parser.add_argument(
    "--resume",
    action="store_true",
    help="if true, tries to resume training from existing checkpoint",
)
parser.add_argument("--load_from", default=None, help="""Path to load checkpoints to resume training.""")
parser.add_argument(
    "--eps",
    default=64,
    type=float,
    help="radius of PGD (Projected Gradient Descent) attack",
)
parser.add_argument(
    "--num-steps",
    default=4,
    type=int,
    help="number of steps of PGD (Projected Gradient Descent) attack",
)
parser.add_argument("--lbd", default=1.0, type=float, help="strength of the contribution of L^MAdv")
parser.add_argument(
    "--eps_double",
    action="store_true",
    help="if true, attack radius (epsilon) is doubled after warmup",
)
parser.add_argument(
    "--warmup_eps",
    default=10000,
    type=int,
    help="after given epoch, the attack radius will be modified",
)

#####################
# ViT Configuration Options Based on iBOT (Zhou et al., 2022) and MAE (Kaiming He et al., 2021)
parser.add_argument(
    "--layer_decay",
    type=float,
    default=0.65,
    help="layer-wise lr decay from ELECTRA/BEiT",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.04,
    help="""Initial value of the
    weight decay. With ViT, a smaller value at the beginning of training works well.""",
)
parser.add_argument(
    "--weight_decay_end",
    type=float,
    default=0.4,
    help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""",
)
parser.add_argument(
    "--use_fp16",
    type=utils.bool_flag,
    default=True,
    help="""Whether or not
    to use half precision for training. Improves training time and memory requirements,
    but can provoke instability and slight decay of performance. We recommend disabling
    mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
)
parser.add_argument(
    "--clip_grad",
    type=float,
    default=0.3,
    help="""Maximal parameter
    gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
    help optimization for larger ViT architectures. 0 for disabling.""",
)
parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate (default: 0.2)")

#####################
# Misc
parser.add_argument(
    "--dist_url",
    default="env://",
    type=str,
    help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""",
)
parser.add_argument(
    "--local-rank",
    default=0,
    type=int,
    help="Please ignore and do not set this argument.",
)

#####################
# Options of FT-CADIS
parser.add_argument(
    "--ft_method",
    choices=["lora", "full-ft"],
    default="full-ft",
    help="choose fine-tuning method",
)
parser.add_argument("--num_noises", type=int, default=4, help="number of noises per sample")
parser.add_argument(
    "--warm_start",
    action="store_true",
    help="if true, at least one denoised image per sample will be optimized",
)

args = parser.parse_args()
args.eps /= 256.0


def main():
    utils.init_distributed_mode(args)
    # fix the seed for reproducibility
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Set the effective batch size considering the number of denoised images per sample
    eff_batch_size = int(args.batch * args.accum_iter * utils.get_world_size() / args.num_noises)
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    if not args.resume:
        args.outdir = f"logs/ft-cadis/{args.ft_method}/{args.dataset}/adv_{args.eps}_{args.num_steps}/lbd_{args.lbd}/num_{args.num_noises}/noise_{args.train_noise_sd}/lr:{args.lr}_eff_batch:{eff_batch_size}"
        args.outdir = args.outdir + f"/{args.id}/"
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir, exist_ok=True)
    else:
        args.outdir = os.path.dirname(args.load_from)

    pin_memory = (args.dataset == "imagenet")
    train_dataset = get_dataset(args.dataset, "train")
    print(f"Train Data loaded: there are {len(train_dataset)} images.")

    test_dataset = get_dataset(args.dataset, "test")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    classifier = get_architecture(args.arch, args.ft_method, args.drop_path)

    print(
        "Number of training parameters: ",
        sum(p.numel() for p in classifier.parameters() if p.requires_grad),
    )

    attacker = PGD(steps=args.num_steps, device=device, max_norm=args.eps)

    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], find_unused_parameters=False)

    if args.dataset == "cifar10":
        denoiser = nn.parallel.DistributedDataParallel(
            CIFAR10_Denoiser(), device_ids=[args.gpu], find_unused_parameters=False
        )
    elif args.dataset == "imagenet":
        denoiser = nn.parallel.DistributedDataParallel(
            ImageNet_Denoiser(), device_ids=[args.gpu], find_unused_parameters=False
        )

    # Get the timestep t corresponding to sigma (set the noise level to twice the original since the range is [-1,1])
    target_sigmas = {"train": args.train_noise_sd * 2, "test": args.test_noise_sd * 2}
    real_sigmas = {"train": 0, "test": 0}
    time_steps = {"train": 0, "test": 0}

    for key in ["train", "test"]:
        while real_sigmas[key] < target_sigmas[key]:
            time_steps[key] += 1
            a = denoiser.module.diffusion.sqrt_alphas_cumprod[time_steps[key]]
            b = denoiser.module.diffusion.sqrt_one_minus_alphas_cumprod[time_steps[key]]
            real_sigmas[key] = b / a

    train_time_step = time_steps["train"]
    test_time_step = time_steps["test"]

    print(f"train_noise level:{args.train_noise_sd}")
    print(f"test_noise level:{args.test_noise_sd}")
    print(f"train diffusion time step:{train_time_step}")
    print(f"test diffusion time step:{test_time_step}")
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    print(f"epsilon:{args.eps}")

    if args.dataset == "cifar10":
        param_groups = lrd.param_groups_lrd(classifier.module, args.weight_decay, layer_decay=args.layer_decay)

    elif args.dataset == "imagenet":
        if args.ft_method == "lora":
            skip_weight_decay_list = classifier.module.classifier.lora_vit.no_weight_decay()
            skip_weight_decay_list = {"classifier.lora_vit." + element for element in skip_weight_decay_list}

        param_groups = lrd.param_groups_lora_lrd(
            classifier.module,
            args.weight_decay,
            skip_weight_decay_list,
            args.layer_decay,
        )

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    criterion = CrossEntropyLoss().to(device)

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(train_loader),
    )

    # print(optimizer)

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    if utils.is_main_process():  # Tensorboard Configuration
        writer = SummaryWriter(args.outdir)

    starting_epoch = 0

    if args.resume:
        model_path = args.load_from

        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            starting_epoch = checkpoint["epoch"]
            classifier.module.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    print("Starting Fine-tuning Off-the-shelf Model!")
    for epoch in range(starting_epoch, args.epochs):
        before = time.time()

        if args.eps_double:
            if epoch >= args.warmup_eps:
                attacker = PGD(steps=args.num_steps, device=device, max_norm=args.eps * 2.0)

        train_loader.sampler.set_epoch(epoch)

        train_stats = ft_cadis(
            train_loader,
            classifier,
            denoiser,
            train_time_step,
            optimizer,
            wd_schedule,
            epoch,
            fp16_scaler,
            attacker,
            device,
        )
        
        if utils.is_main_process():
            writer.add_scalar("loss/sce", train_stats["losses_sce"], epoch)
            writer.add_scalar("loss/madv", train_stats["losses_madv"], epoch)
            writer.add_scalar("loss/total", train_stats["losses_total"], epoch)
            writer.add_scalar("accuracy/train@1", train_stats["top1"], epoch)
            writer.add_scalar("accuracy/train@5", train_stats["top5"], epoch)
            writer.add_scalar("learning_rate/front", train_stats["lr_front"], epoch)
            writer.add_scalar("learning_rate/back", train_stats["lr_back"], epoch)
            writer.add_scalar("weight_decay", train_stats["wd"], epoch)

        test_stats = test(
            test_loader,
            classifier,
            denoiser,
            test_time_step,
            criterion,
            fp16_scaler,
            device,
        )

        if utils.is_main_process():
            writer.add_scalar("loss/test", test_stats["losses"], epoch)
            writer.add_scalar("accuracy/test@1", test_stats["top1"], epoch)
            writer.add_scalar("accuracy/test@5", test_stats["top5"], epoch)

        after = time.time()

        if utils.is_main_process():
            logfilename = os.path.join(args.outdir, "log.txt")

            init_logfile(
                logfilename,
                "epoch\ttime\tlr_front\tlr_back\twd\ttrain loss_sce\ttrain loss_madv\ttrain loss_total\ttrain acc\ttest loss\ttest acc",
            )
            log(
                logfilename,
                "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                    epoch,
                    after - before,
                    train_stats["lr_front"],
                    train_stats["lr_back"],
                    train_stats["wd"],
                    train_stats["losses_sce"],
                    train_stats["losses_madv"],
                    train_stats["losses_total"],
                    train_stats["top1"],
                    test_stats["losses"],
                    test_stats["top1"],
                ),
            )

        if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:

            model_path = os.path.join(args.outdir, f"checkpoint-{epoch+1}.pth.tar")

            save_dict = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": classifier.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            if fp16_scaler is not None:
                save_dict["fp16_scaler"] = fp16_scaler.state_dict()

            torch.save(save_dict, model_path)


def _chunk_minibatch_stochastic(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i * batch_size : (i + 1) * batch_size], y[i * batch_size : (i + 1) * batch_size],


def ft_cadis(
    loader: DataLoader,
    classifier: torch.nn.Module,
    denoiser: torch.nn.Module,
    time_step: int,
    optimizer: Optimizer,
    wd_schedule,
    epoch: int,
    fp16_scaler,
    attacker,
    device: torch.device,
):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    classifier.train()
    denoiser.eval()

    optimizer.zero_grad()

    for i, batch in enumerate(metric_logger.log_every(loader, 10, header)):
        mini_batches = _chunk_minibatch_stochastic(batch, args.num_noises)

        for j, (inputs, targets) in enumerate(mini_batches):
            # we use a per iteration (instead of per epoch) lr scheduler
            if (i * args.num_noises + j) % args.accum_iter == 0:
                global_iter = round((i + j / args.num_noises)) + len(loader) * epoch
                global_iter_eplike = (i + j / args.num_noises) / len(loader) + epoch
                adjust_learning_rate_weight_decay(optimizer, global_iter, global_iter_eplike, wd_schedule, args)

            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            inputs = inputs.repeat(args.num_noises, 1, 1, 1)
            targets_r = targets.repeat(args.num_noises)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # Step 1 : Noise-and-Denoise Procedure
                clean, denoised_inputs = denoiser(inputs, time_step)

                # Step 2 : Confidence-aware Denoised Image Selection & Calculate Consistent Targets for Adversarial Loss
                logits0_c = classifier(denoised_inputs.detach())
                logits0_chunk = torch.chunk(logits0_c, args.num_noises, dim=0)

                confidences = (logits0_c.argmax(1) == targets_r)
                confidences = torch.chunk(confidences, args.num_noises, dim=0)
                confidences = torch.stack(confidences, dim=0)
                confidences = confidences.permute(1, 0).contiguous()

                consistent_targets = F.softmax(logits0_c, dim=1)
                consistent_targets = torch.chunk(consistent_targets, args.num_noises, dim=0)
                consistent_targets = torch.stack(consistent_targets, dim=0)
                consistent_targets = consistent_targets.permute(1, 0, 2).contiguous()
                consistent_targets = consistent_targets.mean(dim=1)

                requires_grad_(args, classifier.module, False)
                classifier.eval()

                # Step 3: Search the Adversarial Examples for the Adversarial Loss
                hard_etas = attacker.attack(classifier, denoised_inputs, consistent_targets, clean)

                classifier.train()
                requires_grad_(args, classifier.module, True)

                # Step 4: Compute the Logit for the Adversarial Loss
                clean = torch.chunk(clean, args.num_noises, dim=0)
                inputs_c = torch.cat([clean[0] + noise for noise in hard_etas], dim=0)
                logits_c = classifier(inputs_c)

                # Step 5 : Compute the Confidence-Aware Selective Cross-Entropy Loss
                loss_sce = [F.cross_entropy(logit, targets, reduction="none").view(-1, 1) for logit in logits0_chunk]
                loss_sce = torch.cat(loss_sce, dim=1)
                loss_sce, loss_index = loss_sce.sort()

                confidences_sorted = torch.gather(confidences, dim=1, index=loss_index)
                if args.warm_start:
                    all_false_rows = torch.all(confidences_sorted == False, dim=1)
                    confidences_sorted[all_false_rows, 0] = True

                loss_sce = (loss_sce * confidences_sorted).mean(1)

                # Step 6: Compute the Confidence-Aware Masked Adversarial Loss
                mask_adv = torch.all(confidences == True, dim=1)

                logits_chunk = torch.chunk(logits_c, args.num_noises, dim=0)

                loss_madv = [
                    F.kl_div(
                        F.log_softmax(logit, dim=1),
                        consistent_targets,
                        reduction="none",
                    ).sum(1, keepdim=True)
                    for logit in logits_chunk
                ]

                loss_madv, _ = torch.cat(loss_madv, dim=1).max(1)

                loss_madv = args.lbd * mask_adv * loss_madv

                # Step 7: Compute the Total Loss
                loss = (loss_sce + loss_madv).mean()

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            loss /= args.accum_iter

            if (i * args.num_noises + j) % args.accum_iter == 0:
                optimizer.zero_grad()

            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if (i * args.num_noises + j + 1) % args.accum_iter == 0:
                    if args.clip_grad:
                        param_norms = utils.clip_gradients(classifier, args.clip_grad)
                    optimizer.step()
            else:  # Use fp16_scaler
                fp16_scaler.scale(loss).backward()
                if (i * args.num_noises + j + 1) % args.accum_iter == 0:
                    if args.clip_grad:
                        fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                        param_norms = utils.clip_gradients(classifier, args.clip_grad)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        acc1, acc5 = accuracy(logits_c, targets_r, topk=(1, 5))
        metric_logger.update(losses_sce=loss_sce.mean().item())
        metric_logger.update(losses_madv=loss_madv.mean().item())
        metric_logger.update(losses_total=loss_value)
        metric_logger.update(top1=acc1.item())
        metric_logger.update(top5=acc5.item())

        if args.arch == "cifar_vit_base":
            metric_logger.update(lr_front=optimizer.param_groups[2]["lr"])
            metric_logger.update(lr_back=optimizer.param_groups[27]["lr"])
            metric_logger.update(wd=optimizer.param_groups[2]["weight_decay"])
        elif args.dataset == "imagenet_vit_base":
            metric_logger.update(lr_front=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr_back=optimizer.param_groups[13]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return return_dict


class PGD(object):
    """
    FT-CADIS PGD attack based on KL-divergence against consistency targets from denoised images

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.

    """

    def __init__(
        self,
        steps: int,
        random_start: bool = True,
        max_norm: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(PGD, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device

    def attack(self, classifier, denoised_inputs, targets, clean):
        """
        Performs FT-CADIS PGD attack of the classifier for the inputs and labels.

        Parameters
        ----------
        classifier : nn.Module
            Classifier to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [-1, 1] range.
        targets : torch.Tensor
            Adversarial targets of the samples to attack.
        clean : torch.Tensor
            Batch of original clean images. Values should be in the [-1, 1] range.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the classifier.

        """
        if denoised_inputs.min() < -1 or denoised_inputs.max() > 1:
            raise ValueError("Input values should be in the [-1, 1] range.")
        if clean.min() < -1 or clean.max() > 1:
            raise ValueError("Original clean images should be in the [-1, 1] range.")

        def _batch_l2norm(x):
            x_flat = x.reshape(x.size(0), -1)
            return torch.norm(x_flat, dim=1)

        m = args.num_noises
        eta0 = denoised_inputs - clean
        eta = eta0.detach()
        batch_size = denoised_inputs.size(0)

        mu0 = eta0.view(batch_size, -1).mean(1).view(-1, 1, 1, 1)
        sigma0 = (eta0**2).view(batch_size, -1).mean(1).sqrt().view(-1, 1, 1, 1)

        # Step size of attack
        alpha = self.max_norm / self.steps * 2
        for _ in range(self.steps):
            eta.requires_grad_()
            logits_r = classifier(clean + eta)

            logits_chunk = torch.chunk(logits_r, m, dim=0)

            loss_kls = [
                F.kl_div(F.log_softmax(logit, dim=1), targets, reduction="none").sum(1) for logit in logits_chunk
            ]
            loss = (sum(loss_kls) / m).sum()

            grad = torch.autograd.grad(loss, [eta])[0]  # returns the sum of gradients of outputs
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

            # Gradient ascent -> Maximize kl loss
            eta = eta + alpha * grad
            diff = eta - eta0
            # Projection based on the l2-norm
            diff = diff.renorm(p=2, dim=0, maxnorm=self.max_norm)

            eta = eta0 + diff

            mu = eta.view(batch_size, -1).mean(1).view(-1, 1, 1, 1)
            sigma = (eta**2).view(batch_size, -1).mean(1).sqrt().view(-1, 1, 1, 1)

            eta = (eta - mu) / sigma
            eta = mu0 + eta * sigma0
            eta = eta.detach()

        return torch.chunk(eta, m, dim=0)


def test(loader, classifier, denoiser, time_step, criterion, fp16_scaler, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # Switch to evaluation mode
    torch.cuda.empty_cache()
    classifier.eval()
    denoiser.eval()

    with torch.no_grad():
        for inputs, targets in metric_logger.log_every(loader, 10, header):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                _, denoised_inputs = denoiser(inputs, time_step)

                outputs = classifier(denoised_inputs)
                loss = criterion(outputs, targets)

            torch.cuda.synchronize()
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            metric_logger.update(losses=loss.item())
            metric_logger.update(top1=acc1.item())
            metric_logger.update(top5=acc5.item())

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return return_dict


if __name__ == "__main__":
    main()
