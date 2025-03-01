from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from fluoressential.style import STYLE


def plot_img(fig_fp, img, cmax=None, show_cbar=False, sbar_microns=None, t_unit=None, regions=None, centroids=None):
    """Plot and annotate a fluorescence microscopy image.

    Args:
        fig_fp (str): absolute filepath for saving the figure
        img (2D array): processed fluorescence image
        cmax (float): max pixel intensity or upper limit of the color scale
        show_cbar (bool): whether to plot a colorbar
        sbar_microns (int): the length in microns equivalent to 200 pixels
            for the scalebar text annotation (specify None for no scalebar)
        t_unit (str): unit for the annotated timestamp
        regions (2D array): binary image for drawing white outlines denoting regions of interest
            TRUE = foreground; FALSE = background
        centroids (dict): {n: (y, x)} coordinates of centroids for annotating ROIs
            with their assigned number
    """
    with sns.axes_style("whitegrid"), mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(24, 16))
        axim = ax.imshow(img, cmap="turbo")
        cmax = np.max(img) if cmax is None else cmax
        axim.set_clim(0.0, cmax)
        if t_unit is not None:
            timepoint = Path(fig_fp).stem
            t_text = timepoint + t_unit
            ax.text(
                0.02,
                0.98,
                t_text,
                ha="left",
                va="top",
                color="white",
                fontsize=STYLE["font.size"],
                weight="bold",
                transform=ax.transAxes,
            )
        if sbar_microns is not None:
            fontprops = font_manager.FontProperties(size=STYLE["font.size"], weight="bold")
            asb = AnchoredSizeBar(
                ax.transData,
                200,
                f"{sbar_microns}\u03bcm",
                color="white",
                size_vertical=20,
                fontproperties=fontprops,
                loc="lower left",
                pad=0,
                borderpad=0.2,
                sep=10,
                frameon=False,
            )
            ax.add_artist(asb)
        if show_cbar:
            cb = fig.colorbar(axim, pad=0.005, format="%.3f", extend="both", extendrect=True, ticks=[0.0, cmax])
            cb.outline.set_linewidth(1)
            cb.ax.tick_params(length=24, width=12, pad=6)
        if regions is not None:
            ax.contour(regions, linewidths=3, colors="w")
            if centroids is not None:
                for num, (y, x) in centroids.items():
                    ax.annotate(
                        str(num),
                        xy=(x, y),
                        xycoords="data",
                        color="white",
                        fontsize=48,
                        ha="center",
                        va="center_baseline",
                    )
        ax.grid(False)
        ax.axis("off")
        fig.canvas.draw()
        fig.savefig(fig_fp, dpi=100)
    plt.close("all")


def plot_bgd(fig_fp, img, bgd):
    """Plot a line profile of the raw image and approximated background.

    Manual quality check for the background subtraction step.

    Args:
        fig_fp (str): absolute filepath for saving the figure
        img (2D array): the raw/unprocessed image before background subtraction
        bgd (2D array): the approx background image
    """
    with sns.axes_style("whitegrid"), mpl.rc_context(STYLE):
        bg_rows = np.argsort(np.var(img, axis=1))[-100:-1:10]
        row_i = np.random.choice(bg_rows.shape[0])
        bg_row = bg_rows[row_i]
        fig, ax = plt.subplots(figsize=(24, 16))
        ax.plot(img[bg_row, :], color="#648FFF")
        ax.plot(bgd[bg_row, :], color="#785EF0")
        fig.savefig(fig_fp, dpi=100)
    plt.close("all")


def plot_ty(fig_fp, ty_df, xlabel, ylabel):
    fig_fp = Path(fig_fp)
    fig_fp.parent.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("whitegrid"), mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(24, 16))
        sns.lineplot(ax=ax, data=ty_df, x="t", y="y", color="#785EF0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=10)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp, dpi=100)
    plt.close("all")
