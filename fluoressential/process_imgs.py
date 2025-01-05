from fluoressential.process import list_img_fps

# def process_group_imgs(group_dp, sb_microns=None, cb_max=None, n_thr=1, gau_scale=1, vert_scale=1, ct_cutoff=0.1):
#     def img_task(img_fp):
#         img, raw, bgd = subtract_bgd(img_fp, n_thr, gau_scale, vert_scale, ct_cutoff)
#         img_fn = os.path.splitext(os.path.basename(img_fp))[0]
#         plot_bgd_prof(os.path.join(res_bgd_dp, str(img_fn) + ".png"), raw, bgd)
#         plot_fluor_img(os.path.join(res_img_dp, str(img_fn) + ".png"), img, cbar_max=cb_max_i, sb_microns=sb_microns)
#         return np.mean(img)

#     data_dp = os.path.join(group_dp, "data")
#     results_dp = os.path.join(group_dp, "results")
#     res_bgd_dp = os.path.join(results_dp, "bgd")
#     makedirs(res_bgd_dp)
#     res_img_dp = os.path.join(results_dp, "img")
#     makedirs(res_img_dp)
#     cb_max_i = cb_max
#     if cb_max is None:
#         cb_max_i = calc_cbar_max(data_dp)
#     y_i = Parallel(n_jobs=os.cpu_count())(delayed(img_task)(img_fp) for img_fp in list_img_files(data_dp))
#     return y_i


def main():
    # group_dp = "/home/phuong/data/proc-data/20241104-fm-k562-coculture/GFP/"
    # process_group_imgs(group_dp, sb_microns=220, cb_max=None, n_thr=1, gau_scale=1, vert_scale=2, ct_cutoff=0.1)
    # print("\a")

    data_dp = "/home/phuong/data/phd-project/1--biosensor/0--ddFP/data/0/imgs/"
    img_fps = list_img_fps(data_dp)
    print(len(img_fps))


if __name__ == "__main__":
    main()
