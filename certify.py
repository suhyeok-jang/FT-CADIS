import os 
import argparse
import time 
import datetime 
from tqdm import tqdm
from torchvision import transforms, datasets

from third_party.smoothed_classifier.core import Smooth 
from architecture import CIFAR10_Denoise_And_Classify, ImageNet_Denoise_And_Classify
from datasets import DATASETS

from utils import utils

from torch.utils.data import Subset

import warnings
warnings.filterwarnings('ignore')

CIFAR10_DATA_DIR = "/data/cifar10/dataset_cache"

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

IMAGENET_LOC_ENV = "IMAGENET_DIR"

# os.environ['IMAGENET_DIR'] = '/home/jsh/ImageNet'
# dir = os.environ['IMAGENET_DIR']
# #Valid dataset 
# subdir = os.path.join(dir, "val")

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
            model = CIFAR10_Denoise_And_Classify(vit_checkpoint = args.finetuned_path)
        elif args.dataset == "imagenet":
            model = ImageNet_Denoise_And_Classify(vit_checkpoint = args.finetuned_path)
        else:
            raise NotImplementedError

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
    elif args.dataset == "imagenet":
        if not IMAGENET_LOC_ENV in os.environ:
            raise RuntimeError("environment variable for ImageNet directory not set")
        
        dir = os.environ[IMAGENET_LOC_ENV]
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        subdir = os.path.join(dir, "val")
        dataset = datasets.ImageFolder(subdir, transform)
        subset_indices = utils.sample_1_per_class_imagenet(dataset.targets) #1000 images
        dataset= Subset(dataset, subset_indices)
        
        
    # Get the timestep t corresponding to noise level sigma x 2 <- [-1,1]
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

    if args.off_the_shelf:
        outfile = f'test/certify_for_paper/imagenet_denoised/off-the-shelf/imagenet_vit_base_patch16_384/noise_{args.sigma}_MC:{args.N}.tsv'
    else:
        outfile = f'test/certify_for_paper/imagenet_denoised/catrs_running_stochastic/noise_0.5/noise_{args.sigma}_MC:{args.N}_{args.finetuned_path.split("/")[-1].replace(".pth.tar","")}.tsv'

    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.exists(outfile):
        raise 'file already exists'
    
    f = open(outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\tright\ttime", file=f, flush=True)

    total_num = 0
    correct = 0
    for i in tqdm(range(len(dataset))):
        if i % args.skip != 0:
            continue

        (x, label) = dataset[i]
        x = x.cuda()

        before_time = time.time()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        after_time = time.time()

        correct += int(prediction == label)
        right = (prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        total_num += 1

        print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{}".format(
            i, label, prediction, radius, correct, right, time_elapsed), file=f, flush=True)

    print("sigma %.2f accuracy of smoothed classifier %.4f "%(args.sigma, correct/float(total_num)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--dataset', type=str, choices=DATASETS)
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=100, help="number of samples to use")
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--finetuned_path", type=str, help="ViT finetuned_path")
    parser.add_argument('--off-the-shelf', action='store_true',
                    help='if true, use only off-the-shelf models')
    args = parser.parse_args()

    main(args)