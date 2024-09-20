# this file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
# written by Jeong et al.
# https://github.com/ethz-spylab/diffusion_denoised_smoothing
# written by Carlini et al.

import os
import argparse
import time
import datetime
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import Subset

from third_party.smoothed_classifier.core import Smooth
from architecture import CIFAR10_Denoise_And_Classify, ImageNet_Denoise_And_Classify
from datasets import get_dataset, DATASETS
from utils import utils

def main(args):
    utils.fix_random_seeds(args.seed)

    if args.off_the_shelf:
        if args.dataset == "cifar10":
            model = CIFAR10_Denoise_And_Classify()
        elif args.dataset == "imagenet":
            model = ImageNet_Denoise_And_Classify()
        else:
            raise NotImplementedError
    else:
        if args.dataset == "cifar10":
            model = CIFAR10_Denoise_And_Classify(vit_checkpoint=args.finetuned_path)
        elif args.dataset == "imagenet":
            model = ImageNet_Denoise_And_Classify(vit_checkpoint=args.finetuned_path)
        else:
            raise NotImplementedError

    dataset = get_dataset(args.dataset, "test")
    if args.dataset == "imagenet":  # Select 1000 images randomly from ImageNet validation set
        subset_indices = utils.sample_1_per_class_imagenet(dataset.targets)
        dataset = Subset(dataset, subset_indices)

    # Get the timestep t corresponding to sigma (noise level) x 2 (Since the image range [-1,1] differs from the typical range [0,1])
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    # Define the smoothed classifier
    if args.dataset == "cifar10":
        smoothed_classifier = Smooth(model, 10, args.sigma, t)
    elif args.dataset == "imagenet":
        smoothed_classifier = Smooth(model, 1000, args.sigma, t)
    else:
        raise NotImplementedError

    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.exists(args.outfile):
        raise "file already exists"

    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tradius\tcorrect_num\tcorrect\ttime", file=f, flush=True)

    total_num = 0
    correct_num = 0
    for i in tqdm(range(len(dataset))):
        if i % args.skip != 0:
            continue

        (x, label) = dataset[i]
        x = x.cuda()

        before_time = time.time()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        after_time = time.time()

        correct_num += int(prediction == label)
        correct = (prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        total_num += 1

        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{}\t{}".format(i, label, prediction, radius, correct_num, correct, time_elapsed),
            file=f,
            flush=True,
        )

    print("sigma %.2f accuracy of smoothed classifier %.4f " % (args.sigma, correct_num / float(total_num)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on many examples")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", type=str, choices=DATASETS)
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=100, help="number of samples to use")
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--finetuned_path", type=str, help="Checkpoint path of the fine-tuned model")
    parser.add_argument(
        "--off-the-shelf",
        action="store_true",
        help="if true, directly use off-the-shelf models (no fine-tuned)",
    )
    parser.add_argument("--outfile", type=str, help="output file")
    args = parser.parse_args()

    main(args)
