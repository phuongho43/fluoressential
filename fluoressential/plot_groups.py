from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns

from fluoressential.process import calc_ave_regimes
from fluoressential.style import PALETTE, STYLE_LARGE


def plot_groups(
    fig_fp, y_csv_fp, group_labels, ylabel, log2_yaxis=False, palette=PALETTE, figsize=(24, 16), rc_params=STYLE_LARGE
):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figsize)
        y_df = pd.read_csv(y_csv_fp)
        sns.stripplot(
            ax=ax,
            data=y_df,
            x="group",
            y="response",
            hue="group",
            palette=palette,
            jitter=0.1,
            size=2 * rc_params["lines.markersize"],
            linewidth=rc_params["lines.markeredgewidth"],
            edgecolor=rc_params["lines.markeredgecolor"],
            alpha=0.8,
            zorder=2.1,
        )
        sns.pointplot(
            ax=ax,
            data=y_df,
            x="group",
            y="response",
            hue="group",
            errorbar=("se", 1.96),
            palette="dark:#212121",
            markers=".",
            err_kws={"linewidth": 10},
            linestyles=None,
            capsize=0.2,
            zorder=2.2,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels)
        if log2_yaxis:
            ax.set_yscale("log", base=2)
        ax.locator_params(axis="y", nbins=10)
        ax.get_legend().remove()
        fig.savefig(fig_fp)
    plt.close("all")


def main():
    fig_fp = "/home/phuong/data/phd-project/figures/fig_2j.png"
    y_csv_fp = "/home/phuong/data/phd-project/1--biosensor/3--sparser/results/y.csv"
    group_labels = [
        "None\nInput",
        "Sparse\nInput",
        "Dense\nInput",
    ]
    palette = ["#34495E", "#EA822C", "#8069EC"]
    ylabel = r"$\mathbf{Ave\ \Delta F/F_{0}}$"
    time_regimes = [[0, 60], [60, 120], [120, 155]]
    ave_regimes_df = calc_ave_regimes(y_csv_fp, time_regimes)
    y_csv_fp = "/home/phuong/data/phd-project/1--biosensor/3--sparser/results/ave_regimes.csv"
    ave_regimes_df.to_csv(y_csv_fp, index=False)
    ave_regimes_df = pd.read_csv(y_csv_fp)
    ave_none = ave_regimes_df.loc[(ave_regimes_df["group"] == 0), ["response"]]
    ave_sparse = ave_regimes_df.loc[(ave_regimes_df["group"] == 1), ["response"]]
    ave_dense = ave_regimes_df.loc[(ave_regimes_df["group"] == 2), ["response"]]
    print(sp.stats.ttest_ind(ave_none, ave_sparse))
    print(sp.stats.ttest_ind(ave_sparse, ave_dense))
    plot_groups(fig_fp, y_csv_fp, group_labels, ylabel=ylabel, palette=palette, rc_params=STYLE_LARGE)


if __name__ == "__main__":
    main()
