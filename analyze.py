# this file is based on code publicly available at
# https://github.com/alinlab/smoothing-catrs
# written by Jeong et al.

from typing import *
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):  # Default Metric
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):  # Approximate certified test accuracy at radius r
        return (df["correct"] & (df["radius"] >= radius)).mean()

    def acr(self):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return (df["correct"] * df["radius"]).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (
            mean
            - self.alpha
            - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
            - math.log(1 / self.rho) / (3 * num_examples)
        )


class Line(object):
    def __init__(
        self,
        quantity,
        legend: str,
        scale_x: float = 1.0,
        linewidth: float = 1.0,
        linestyle: str = "-",
    ):
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


def plot_certified_accuracy(
    outfile: str,
    title: str,
    min_radius: float,
    max_radius: float,
    lines: List[Line],
    radius_step: float = 0.01,
    min_y=None,
    max_y=None,
) -> None:
    radii = np.arange(min_radius, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        # plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)
        plt.plot(
            radii * line.scale_x,
            line.quantity.at_radii(radii),
            linewidth=line.linewidth,
            linestyle=line.linestyle,
        )

    plt.ylim((min_y if min_y is not None else 0.0, max_y if max_y is not None else 1.0))
    plt.xlim((min_radius, max_radius))
    # Set a custom xticks range
    plt.xticks(np.arange(0.5, 2.0, 0.5))
    plt.tick_params(labelsize=14)
    plt.xlabel(r"Radius ($\ell_2$)", fontsize=16)
    plt.ylabel("Certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc="upper right", fontsize=11)
    # plt.legend([method.legend for method in lines], loc ='lower left', fontsize=12)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def smallplot_certified_accuracy(
    outfile: str,
    title: str,
    max_radius: float,
    methods: List[Line],
    radius_step: float = 0.01,
    xticks=0.5,
) -> None:
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
    plt.legend([method.legend for method in methods], loc="upper right", fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy_acr(
    outfile: str,
    radius_start: float,
    radius_stop: float,
    radius_step: float,
    methods: List[Line],
):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)
        acr = method.quantity.acr()

    f = open(outfile, "w")

    for i, radius in enumerate(radii):
        if i == 0:
            f.write("ACR & $r = {:.3}$".format(radius))
        else:
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
            txt = " & {:.1f}".format(accuracies[i, j] * 100)
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(
    outfile: str,
    radius_start: float,
    radius_stop: float,
    radius_step: float,
    methods: List[Line],
):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, "w")
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


def pretty_table_certified_accuracy_ACR(
    outfile: str,
    radius_start: float,
    radius_stop: float,
    radius_step: float,
    methods: List[Line],
):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    table = PrettyTable()
    field_names = ["Radius"]
    field_names.extend([f"r = {radius:.2f}" for radius in radii])
    field_names.extend(["ACR"])
    table.field_names = field_names

    for method in methods:
        accuracies = method.quantity.at_radii(radii)
        acr = method.quantity.acr()
        row = [method.legend]
        row.extend([f"{accuracy:.3f}" for accuracy in accuracies])
        row.extend([f"{acr:.3f}"])
        table.add_row(row)

    print(table)

    with open(outfile, "w") as f:
        f.write(table.get_string())
