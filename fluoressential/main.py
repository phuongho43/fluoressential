import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from natsort import natsorted
from skimage import img_as_float
from skimage.io import imread

from fluoressential.plot import plot_bgd, plot_img
from fluoressential.process import calc_imgs_cmax, list_img_fps, subtract_bgd


def img_task(img_fp, results_dp, sub_bgd_kwargs, plot_img_kwargs):
    img_fp = Path(img_fp)
    img_fn = img_fp.stem
    img = img_as_float(imread(img_fp))
    raw = img.copy()
    img, bgd = subtract_bgd(img, **sub_bgd_kwargs)
    results_dp = Path(results_dp)
    res_imgs_dp = results_dp / "imgs"
    res_imgs_dp.mkdir(parents=True, exist_ok=True)
    res_bgds_dp = results_dp / "bgds"
    res_bgds_dp.mkdir(parents=True, exist_ok=True)
    plot_img(res_imgs_dp / f"{img_fn}.png", img, **plot_img_kwargs)
    plot_bgd(res_bgds_dp / f"{img_fn}.png", raw, bgd)
    ti = str(img_fn)
    yi = np.mean(img)
    return {"t": ti, "y": yi}


def analyze_imgs(rep_dp, n_thr=2, gau_scale=1, vert_scale=2, ct_cutoff=0.1, cmax=None, show_cbar=False, sbar_microns=None, t_unit=None):
    rep_dp = Path(rep_dp)
    dat_imgs_dp = rep_dp / "imgs"
    results_dp = rep_dp / "results"
    cmax = calc_imgs_cmax(dat_imgs_dp) if cmax is None else cmax
    sub_bgd_kwargs = {"n_thr": n_thr, "gau_scale": gau_scale, "vert_scale": vert_scale, "ct_cutoff": ct_cutoff}
    plot_img_kwargs = {"cmax": cmax, "show_cbar": show_cbar, "sbar_microns": sbar_microns, "t_unit": t_unit}
    data = Parallel(n_jobs=max(1, os.cpu_count() - 1))(delayed(img_task)(img_fp, results_dp, sub_bgd_kwargs, plot_img_kwargs) for img_fp in list_img_fps(dat_imgs_dp))
    df = pd.DataFrame(data)
    y_csv_fp = results_dp / "y.csv"
    df.to_csv(y_csv_fp, index=False)


def main():
    ## Analyze Single Image ##
    ## Figure 2B ##
    img_fp = "/home/phuong/data/phd-project/1--biosensor/6--examples/1--LOVfast/1/imgs/60.1.tiff"
    results_dp = "/home/phuong/data/phd-project/1--biosensor/6--examples/1--LOVfast/1/results"
    sub_bgd_kwargs = {"n_thr": 2, "gau_scale": 6, "vert_scale": 2, "ct_cutoff": 0.05}
    plot_img_kwargs = {"cmax": 0.03, "show_cbar": False, "sbar_microns": 22, "t_unit": None}
    img_task(img_fp, results_dp, sub_bgd_kwargs, plot_img_kwargs)
    img_fp = "/home/phuong/data/phd-project/1--biosensor/6--examples/1--LOVfast/1/imgs/61.9.tiff"
    plot_img_kwargs = {"cmax": 0.03, "show_cbar": True, "sbar_microns": None, "t_unit": None}
    img_task(img_fp, results_dp, sub_bgd_kwargs, plot_img_kwargs)
    ## Figure 2E ##
    img_fp = "/home/phuong/data/phd-project/1--biosensor/6--examples/3--iLIDfast/0/imgs/60.1.tiff"
    results_dp = "/home/phuong/data/phd-project/1--biosensor/6--examples/3--iLIDfast/0/results"
    sub_bgd_kwargs = {"n_thr": 2, "gau_scale": 6, "vert_scale": 2, "ct_cutoff": 0.05}
    plot_img_kwargs = {"cmax": 0.02, "show_cbar": False, "sbar_microns": 22, "t_unit": None}
    img_task(img_fp, results_dp, sub_bgd_kwargs, plot_img_kwargs)
    img_fp = "/home/phuong/data/phd-project/1--biosensor/6--examples/3--iLIDfast/0/imgs/61.9.tiff"
    plot_img_kwargs = {"cmax": 0.02, "show_cbar": True, "sbar_microns": None, "t_unit": None}
    img_task(img_fp, results_dp, sub_bgd_kwargs, plot_img_kwargs)

    ## ddFP Biosensor Dynamics ##
    expt_dps = [
        "/home/phuong/data/phd-project/1--biosensor/0--ddFP/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/1--LOV/1--V416I/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/0--BL20uW/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/1--BL200uW/",
        "/home/phuong/data/phd-project/1--biosensor/2--intensity/2--BL2000uW/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/3--iLID/1--V416I/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/0--13AA/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/0--13AA/1--V416I/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/1--20AA/0--I427V/",
        "/home/phuong/data/phd-project/1--biosensor/4--linker/1--20AA/1--V416I/",
        "/home/phuong/data/phd-project/1--biosensor/5--decoder/0--sparse-ddFP/",
    ]
    for expt_dp in [Path(expt_dp) for expt_dp in expt_dps]:
        for rep_dp in [dp for dp in natsorted(expt_dp.glob("*")) if dp.is_dir()]:
            print(rep_dp)
            analyze_imgs(rep_dp, n_thr=2, gau_scale=6, vert_scale=-1, ct_cutoff=0.05, cmax=None, show_cbar=True, sbar_microns=22, t_unit="s")

    ## RFP Expression Groups ##
    expt_dps = [
        "/home/phuong/data/phd-project/3--expression/0--HEK-BL-intensity/",
        "/home/phuong/data/phd-project/3--expression/1--HEK-FM_single/",
        "/home/phuong/data/phd-project/3--expression/2--HEK-FM_dual/",
        "/home/phuong/data/phd-project/3--expression/3--K562-FM_single/",
    ]
    for expt_dp in [Path(expt_dp) for expt_dp in expt_dps]:
        for class_dp in [dp for dp in natsorted(expt_dp.glob("*")) if dp.is_dir()]:
            for group_dp in [dp for dp in natsorted(class_dp.glob("*")) if dp.is_dir()]:
                for rep_dp in [dp for dp in natsorted(group_dp.glob("*")) if dp.is_dir()]:
                    print(rep_dp)
                    analyze_imgs(rep_dp, n_thr=2, gau_scale=1, vert_scale=2, ct_cutoff=0.1, cmax=None, show_cbar=True, sbar_microns=220, t_unit=None)


if __name__ == "__main__":
    main()
