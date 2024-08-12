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
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01, min_y = None) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        accuracies = line.quantity.at_radii(radii)
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)
        
        # 반경과 정확도를 데이터프레임으로 변환하여 저장
        result_df = pd.DataFrame({'radius': radii * line.scale_x, 'accuracy': accuracies})
        output_tsv_path = outfile + f"_{line.legend.replace(' ', '_')}.tsv"
        result_df.to_csv(output_tsv_path, sep='\t', index=False)
        print(f"Saved data for {line.legend} to {output_tsv_path}")
    
    if min_y is not None:
        plt.ylim((min_y,1.0))
    else:
        plt.ylim((0.,1.))
    plt.xlim((0, max_radius))
    plt.xticks(np.arange(0,max_radius+0.5,0.5))
    plt.tick_params(labelsize=14)
    plt.xlabel(r"Radius ($\ell_2$)", fontsize=16)
    plt.ylabel("Certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
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

    ablation_1 = '/home/jsh/code_for_cifar10/test/certify/cifar10_ablation/noise_0.5_MC:100000_lbd_0.5_checkpoint-19.tsv'
    ablation_2 = '/home/jsh/code_for_cifar10/test/certify/cifar10_ablation/noise_0.5_MC:100000_lbd_1.0_checkpoint-19.tsv'
    ablation_3 = '/home/jsh/code_for_cifar10/test/certify/cifar10_ablation/noise_0.5_MC:100000_lbd_2.0_checkpoint-19.tsv'
    ablation_4 = '/home/jsh/code_for_cifar10/test/certify/cifar10_ablation/noise_0.5_MC:100000_lbd_4.0_checkpoint-19.tsv'
    ablation_5 = '/home/jsh/code_for_cifar10/test/certify/cifar10_ablation/noise_0.5_MC:100000_lbd_8.0_checkpoint-19.tsv'
    
    ablation_1_acc = ApproximateAccuracy(ablation_1)
    ablation_2_acc = ApproximateAccuracy(ablation_2)
    ablation_3_acc = ApproximateAccuracy(ablation_3)
    ablation_4_acc = ApproximateAccuracy(ablation_4)
    ablation_5_acc = ApproximateAccuracy(ablation_5)

    # title = f"Certified Test Accuracy on CIFAR-10"
    
    max_radius = 2.0
    min_y = -0.05
    
    # 반경 범위와 스텝 설정
    radius_start = 0.0
    radius_stop = 2
    radius_step = 0.25

    # outfile_txt_1_root = os.path.join(os.path.dirname(challenger1_path), 'results')
    outfile_png_root = os.path.join(os.path.dirname(ablation_1), 'plots')
    # os.makedirs(outfile_txt_1_root,exist_ok=True)
    os.makedirs(outfile_png_root,exist_ok=True)

    # outfile_txt_1_path = os.path.join(outfile_txt_1_root, f'{os.path.basename(challenger1_path).replace(".tsv","")}_evaluation.txt')

    outfile_png_path = os.path.join(outfile_png_root, f'effect_of_lambda')

    lines = [Line(quantity=ablation_1_acc, legend=r"$lambda$ = 0.5"),
             Line(quantity=ablation_2_acc, legend=r"$lambda$ = 1.0"),
             Line(quantity=ablation_3_acc, legend=r"$lambda$ = 2.0"),
             Line(quantity=ablation_4_acc, legend=r"$lambda$ = 4.0"),
             Line(quantity=ablation_5_acc, legend=r"$lambda$ = 8.0")]
    
    # #latex table생성
    # latex_table_certified_accuracy_acr(outfile_txt_1_path, radius_start, radius_stop, radius_step, [Line(quantity=challenger1_accuracy, legend="Certificated Accuracy")])

    # Call the plotting function
    plot_certified_accuracy(outfile=outfile_png_path, title=None, max_radius=max_radius, min_y = min_y, lines=lines)