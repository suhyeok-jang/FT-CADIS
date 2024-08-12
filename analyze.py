# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

from typing import *
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable

sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy): #Default Metric
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        #Approximate certified test accuracy at radius r
        # return (df["correct"] & (df["radius"] >= radius)).mean()
        return (df["right"] & (df["radius"] >= radius)).mean()

    def acr(self):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        #ACR 구하기
        # return (df["correct"] * df["radius"]).mean()
        return (df["right"] * df["radius"]).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float): #Not Approixmate certified, 조금 다름
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
    def __init__(self, quantity, legend: str, scale_x: float = 1.0, linewidth: float = 1.0, linestyle: str = '-'):
        self.scale_x = scale_x
        self.quantity = quantity
        # self.plot_fmt = plot_fmt
        self.legend = legend
        self.linewidth = linewidth
        self.linestyle = linestyle
        
class Quantity:
    def __init__(self, values: List[float], radii: List[float]):
        self.values = values
        self.radii = radii

    def at_radii(self, query_radii: np.ndarray) -> np.ndarray:
        return np.interp(query_radii, self.radii, self.values)


def plot_certified_accuracy(outfile: str, title: str, min_radius :float, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01, min_y = None, max_y = None) -> None:
    radii = np.arange(min_radius, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        # plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii),
                 linewidth=line.linewidth, linestyle=line.linestyle)
    
    if min_y is not None and max_y is not None:
        plt.ylim((min_y, max_y))
    elif min_y is not None:
        plt.ylim((min_y,1.0))
    elif max_y is not None:
        plt.ylim((0., max_y))
    else:
        plt.ylim((0.,1.))
    plt.xlim((min_radius, max_radius))
    plt.xticks(np.arange(0.5,2.0,0.5))
    plt.tick_params(labelsize=14)
    plt.xlabel(r"Radius ($\ell_2$)", fontsize=16)
    plt.ylabel("Certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=11)
    # plt.legend([method.legend for method in lines], loc ='lower left', fontsize=12)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()

# def plot_trade_off(outfile: str, title: str, lines: List[Line], min_y = None) -> None:
#     plt.figure()
#     for line in lines:
#         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)
    
#     if min_y is not None:
#         plt.ylim((min_y,1.0))
#     else:
#         plt.ylim((0.,1.))
#     plt.xlim((0, max_radius))
#     plt.tick_params(labelsize=14)
#     plt.xlabel("Radius", fontsize=16)
#     plt.ylabel("Certified accuracy", fontsize=16)
#     # plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
#     plt.legend([method.legend for method in lines], loc ='lower left', fontsize=16)
#     plt.savefig(outfile + ".pdf")
#     plt.tight_layout()
#     plt.title(title, fontsize=16)
#     plt.tight_layout()
#     plt.savefig(outfile + ".png", dpi=300)
#     plt.close()


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy_acr(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)
        acr = method.quantity.acr()

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        f.write(" & {:.3f}".format(acr))
        for j, radius in enumerate(radii):
            # if i == accuracies[:, j].argmax():
            #     txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            # else:
            txt = " & {:.1f}".format(accuracies[i, j]*100)
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.2f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.2f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()

#ACR + Approximate Certificated Test Accuracy 한번에
def pretty_table_certified_accuracy_ACR(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                    methods : List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    table = PrettyTable()
    field_names = ["Radius"]
    field_names.extend([f"r = {radius:.2f}" for radius in radii])
    field_names.extend(['ACR'])
    table.field_names = field_names

    for method in methods:
        accuracies = method.quantity.at_radii(radii)
        acr = method.quantity.acr()
        row = [method.legend]
        row.extend([f"{accuracy:.3f}" for accuracy in accuracies])
        row.extend([f"{acr:.3f}"])
        table.add_row(row)

    print(table)
    # If you want to save the table to a file
    with open(outfile, 'w') as f:
        f.write(table.get_string())


if __name__ == "__main__":
    # Define outfile and other parameters
    ablation_1 = '/home/jsh/code_for_cifar10/test/certify/cifar_ablation_1.00_real/catrs_ablation/final_noise_1.0_MC:100000.tsv'
    ablation_2 = '/home/jsh/code_for_cifar10/test/certify/cifar_ablation_1.00_real/catrs_nomask_ablation/final_noise_1.0_MC:100000.tsv'
    ablation_3 = '/home/jsh/code_for_cifar10/test/certify/cifar_ablation_1.00_real/gaussian_ablation/final_noise_1.0_MC:100000.tsv'
    ablation_4 = '/home/jsh/code_for_cifar10/test/certify/cifar_ablation_1.00_real/gaussian_adv_masking/final_noise_1.0_MC:100000.tsv'
    # ablation_5 = '/home/jsh/code_for_cifar10/test/certify/catrs_ablation_0.50/noise_0.5_MC:10000_M:4_catrs_ablation_og_checkpoint-19.tsv'
    
    ablation_1_acc = ApproximateAccuracy(ablation_1)
    ablation_2_acc = ApproximateAccuracy(ablation_2)
    ablation_3_acc = ApproximateAccuracy(ablation_3)
    ablation_4_acc = ApproximateAccuracy(ablation_4)
    # ablation_5_acc = ApproximateAccuracy(ablation_5)

    # noise_path = '/home/jsh/smoothing-catrs/test/certify_for_paper/imagenet_denoised/catrs_running_stochastic/noise_1.0/noise_1.0_MC:10000_checkpoint-4.tsv'
    # noise_accuracy = ApproximateAccuracy(noise_path)

    # title = f"Certified Test Accuracy on CIFAR-10"
    
    min_radius = 0.2
    max_radius = 1.5
    max_y = None
    min_y = -0.05
    
    # 반경 범위와 스텝 설정
    radius_start = 0.0
    radius_stop = 2.0
    radius_step = 0.25

    outfile_txt_1_root = os.path.join(os.path.dirname(ablation_1), 'results')
    outfile_txt_2_root = os.path.join(os.path.dirname(ablation_2), 'results')
    outfile_txt_3_root = os.path.join(os.path.dirname(ablation_3), 'results')
    outfile_txt_4_root = os.path.join(os.path.dirname(ablation_4), 'results')
    # outfile_png_root = os.path.join(os.path.dirname(noise_path), 'plots')
    os.makedirs(outfile_txt_1_root,exist_ok=True)
    os.makedirs(outfile_txt_2_root,exist_ok=True)
    os.makedirs(outfile_txt_3_root,exist_ok=True)
    os.makedirs(outfile_txt_4_root,exist_ok=True)
    # os.makedirs(outfile_png_root,exist_ok=True)

    outfile_txt_1_path = os.path.join(outfile_txt_1_root, f'{os.path.basename(ablation_1).replace(".tsv","")}_evaluation.txt')
    outfile_txt_2_path = os.path.join(outfile_txt_2_root, f'{os.path.basename(ablation_2).replace(".tsv","")}_evaluation.txt')
    outfile_txt_3_path = os.path.join(outfile_txt_3_root, f'{os.path.basename(ablation_3).replace(".tsv","")}_evaluation.txt')
    outfile_txt_4_path = os.path.join(outfile_txt_4_root, f'{os.path.basename(ablation_4).replace(".tsv","")}_evaluation.txt')
    # outfile_txt_5_path = os.path.join(outfile_txt_1_root, f'{os.path.basename(ablation_5).replace(".tsv","")}_evaluation.txt')

    
    # outfile_png_path = os.path.join(outfile_png_root, f'certified_accuracy_ds')

    # lines = [Line(quantity=noise_small_accuracy, legend=r"$\sigma$ = 0.25"),
    #         Line(quantity=noise_medium_accuracy, legend=r"$\sigma$ = 0.50"),
    #         Line(quantity=noise_large_accuracy, legend=r"$\sigma$ = 1.00")]
    
    # # 데이터
    # r_values = [0.5, 1.0, 1.5, 2.0]
    # certified_accuracy_carlini = [0.711, 0.543, 0.381, 0.295]
    # certified_accuracy_multi_scale = [0.546, 0.398,0.230,0.146]
    # certified_accuracy_ours = [0.719, 0.601, 0.458, 0.394]  
    
    # lee_r_values = [0.5, 1.0, 1.5]
    # certified_accuracy_denoised = [0.330,0.140,0.060]
    # certified_accuracy_lee = [0.410,0.240,0.110]
    
    
    # r_values = [0.25, 0.50, 0.75, 1.00]
    # certified_accuracy_denoised = [0.560,0.410,0.280,0.190]
    # certified_accuracy_lee = [0.600,0.420,0.280,0.190]
    # certified_accuracy_carlini = [0.793, 0.655, 0.487, 0.355]
    # certified_accuracy_ours = [0.803,0.684,0.545,0.399] 
    
    # multi_r_values = [0.50, 1.00]
    # certified_accuracy_multi_scale = [0.619,0.329]
    
    
    
    # denoised_quantity = Quantity(certified_accuracy_denoised, lee_r_values)
    # lee_quantity = Quantity(certified_accuracy_lee, lee_r_values)
    # carlini_quantity = Quantity(certified_accuracy_carlini, r_values)
    # ours_quantity = Quantity(certified_accuracy_ours, r_values)
    # multi_scale_quantity = Quantity(certified_accuracy_multi_scale, r_values)
    
    # line = [Line(quantity= ours_quantity, legend='Ours',linewidth=2.0, linestyle='--'),
    #         Line(quantity= denoised_quantity, legend='Salman et al. (2020)'),
    #         Line(quantity= lee_quantity, legend='Lee. (2021)'),
    #         Line(quantity= carlini_quantity, legend='Carlini et al. (2022)'),
    #         Line(quantity= multi_scale_quantity, legend = 'Jeong et al. (2024)')]
    
    
    # #latex table생성
    latex_table_certified_accuracy_acr(outfile_txt_1_path, radius_start, radius_stop, radius_step, [Line(quantity=ablation_1_acc, legend="Certificated Accuracy")])
    latex_table_certified_accuracy_acr(outfile_txt_2_path, radius_start, radius_stop, radius_step, [Line(quantity=ablation_2_acc, legend="Certificated Accuracy")])
    latex_table_certified_accuracy_acr(outfile_txt_3_path, radius_start, radius_stop, radius_step, [Line(quantity=ablation_3_acc, legend="Certificated Accuracy")])
    latex_table_certified_accuracy_acr(outfile_txt_4_path, radius_start, radius_stop, radius_step, [Line(quantity=ablation_4_acc, legend="Certificated Accuracy")])
    # latex_table_certified_accuracy_acr(outfile_txt_5_path, radius_start, radius_stop, radius_step, [Line(quantity=ablation_5_acc, legend="Certificated Accuracy")])

    # Call the plotting function
    # plot_certified_accuracy(outfile=outfile_png_path, title=None, min_radius = min_radius, max_radius=max_radius, min_y = min_y, max_y = max_y, lines=line)