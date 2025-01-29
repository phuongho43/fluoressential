import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
from matplotlib import patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from fluoressential.style import STYLE


def plot_img(fig_fp, img, cbar_max=None, sbar_microns=None, t_unit=None, note_stim=False, regions=None, centroids=None):
    """Plot and annotate a fluorescence microscopy image.

    Args:
        fig_fp (str): absolute filepath for saving the figure
        img (2D array): processed fluorescence image
        cbar_max (float): maximum value for the colormap scale and colorbar
        sbar_microns (int): the length in microns equivalent to 200 pixels
            for the scalebar text annotation (specify None for no scalebar)
        t_unit (str): unit for the annotated timestamp
        note_stim (bool): whether to draw a blue outline around the whole image to denote
            input signal/stimuli exposure
        regions (2D array): binary image for drawing white outlines denoting regions of interest
            TRUE = foreground; FALSE = background
        centroids (dict): {n: (y, x)} coordinates of centroids for annotating ROIs
            with their assigned number
    """
    with sns.axes_style("whitegrid"), mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(24, 16))
        axim = ax.imshow(img, cmap="turbo")
        axim.set_clim(0.0, cbar_max)
        if t_unit is not None:
            timepoint = os.path.splitext(os.path.basename(fig_fp))[0]
            t_text = "t = " + timepoint + t_unit
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
        if note_stim:
            w, h = img.shape
            ax.add_patch(mpatches.Rectangle((2, 2), w - 7, h - 7, linewidth=10, edgecolor="#648FFF", facecolor="none"))
        cb = fig.colorbar(axim, pad=0.005, format="%.3f", extend="both", extendrect=True, ticks=[0.0, cbar_max])
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
        fig, ax = plt.subplots(figsize=(24, 20))
        ax.plot(img[bg_row, :], color="#648FFF")
        ax.plot(bgd[bg_row, :], color="#785EF0")
        fig.savefig(fig_fp, dpi=100)
    plt.close("all")
