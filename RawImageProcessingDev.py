# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: Rep 1-3
# jupyter: python3
# ---

# %%
import warnings
warnings.filterwarnings("ignore")
import math, os, sys, copy, pickle
from pathlib import Path
home = str(Path.home())
from skimage import io, filters, restoration
from skimage import morphology as morph
import PIL
from PIL import Image, ImageSequence, ImageEnhance
import numpy as np
import pandas as pd
# Make images higher resolution and set default size
import matplotlib
# %matplotlib inline
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (6, 6)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import cv2
import multipagetiff as mtif
import tifffile
import plotly.express as px


# %%
def plot(tif_files, try_denoise = False):
    fig, axes = plt.subplots(2, 4, figsize=(16,8), sharex=True, sharey=True)
    for num, tif_file in enumerate(tif_files):
        #print(tif_file, tif_file.rsplit(".", 1)[0][-1])
        # num = int(tif_file.rsplit(".", 1)[0][-1])-1
        im = Image.open(tif_file)
        if denoise:
            imarray = denoise(np.array(im))
        else:
            imarray = np.array(im)
        # print(num, num//4, num%4)
        axes[num//4, num%4].imshow(imarray)
        axes[num//4, num%4].set_title("{}".format(num+1))
    plt.show()
    # plt.colorbar()
#     fig.subplots_adjust(right=0.85)
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
def plot_oneZ(tif_files, z_pos = 0, z_num = 0):
    fig, axes = plt.subplots(2, 4, figsize=(16,8), sharex=True, sharey=True)
    for tif_file in tif_files:
        #print(tif_file, tif_file.rsplit(".", 1)[0][-1])
        num = int(tif_file.rsplit(".", 1)[0][-1])-1
        imgs = mtif.read_stack(tif_file)
        if z_num != 0:
            img = imgs[z_num - 1]
        else:
            img = imgs[math.floor(len(imgs) * z_pos)]
        # print(num, num//4, num%4)
        axes[num//4, num%4].imshow(img)
        axes[num//4, num%4].set_title("{}".format(num+1))
    plt.show()
def plot_single(img):
    #print(tif_file, tif_file.rsplit(".", 1)[0][-1])
    # im = Image.open(tif_file)
    # img = np.array(im)
    plt.figure()
    ax = plt.gca()
    im_vis = ax.imshow(img)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_vis, cax=cax)
    plt.show()
def flatten_plot(tif_files):
    # fig, axes = plt.subplots(2,4, figsize=(16,8), sharex = True, sharey = True)
    plt.figure(figsize=(16,8))
    for i, tif_file in enumerate(tif_files):
        imgs = mtif.read_stack(tif_file)
        plt.subplot(2,4,i+1)
        img = mtif.flatten(imgs)
        plt.imshow(img)
    plt.tight_layout()

def normalize(img, min_range = 32, im_max = None, im_min = None, scale_to = 65535):
    img_new = copy.deepcopy(img)
    if im_max == None:
        im_max = np.max(img)
    if im_min == None:
        im_min = np.min(img)
    img_new = (img - im_min) / max([(im_max - im_min), min_range])
    img_new = np.array(img_new * scale_to).astype(np.uint16)
    return(img_new)
def filter(img, min_thr = 0, threshold = None):
    if threshold == None:
        try:
            threshold = max(filters.threshold_otsu(img), min_thr)
        except:
            threshold = img.min() + min_thr
    mask = img > threshold
    otsu_filtered = np.zeros_like(img)
    otsu_filtered[mask] = img[mask]
    return(otsu_filtered)
def filter_z(imgz, min_thr = 0, threshold = None):
    for i in range(len(imgz)):
        imgz[i] = filter(imgz[i], min_thr, threshold)
    return(imgz)
def sub_bg_thr(img_ori, min_thr = 0, threshold = None):
    img = copy.deepcopy(img_ori)
    if threshold == None:
        try:
            threshold = max(filters.threshold_otsu(img), min_thr)
        except:
            threshold = img.min() + min_thr
    img[img < threshold] = threshold
    img = img - threshold
    return(img)
def sub_bg_thr_z(imgz, min_thr = 0, threshold = None):
    for i in range(len(imgz)):
        imgz[i] = sub_bg_thr(imgz[i], min_thr, threshold)
    return(imgz)
def threshold_z(imgz):
    thresholds = [filters.threshold_otsu(img) for img in imgz]
    return(max(thresholds))
def threshold_all(imgs):
    thresholds = [max([filters.threshold_otsu(img) for img in imgz]) for imgz in imgs]
    return(max(thresholds)) 
def remove_background(img, rolling_ball_radius = 10):
    background = restoration.rolling_ball(img, radius=rolling_ball_radius)
    rolling_ball_filtered = img - background
    return(rolling_ball_filtered)
def remove_background_z(imgz, rolling_ball_radius = 10):
    for i in range(len(imgz)):
        imgz[i] = remove_background(imgz[i], rolling_ball_radius)
    return(imgz)
def denoise(img, min_thr = 0, rolling_ball_filtered = 10, threshold = None):
    return(filter(remove_background(img, rolling_ball_filtered), min_thr, threshold))
def denoise_z(imgz, min_thr = 0, rolling_ball_radius = 10, threshold = None):
    imgz_denoised = copy.deepcopy(imgz)
    try:
        for i in range(len(imgz)):
            imgz_denoised[i] = remove_background(imgz[i], rolling_ball_radius)
        if threshold == None:
            threshold = threshold_z(imgz_denoised)
        for i in range(len(imgz)):
            imgz_denoised[i] = filter(imgz_denoised[i], min_thr, threshold)
    except:
        # imgz_denoised = normalize(imgz)
        print("Denoising failed, returning the original image")
    return(imgz_denoised)

def max_z(imgz):
    try:
        img_max = np.max(imgz, axis=0)
    except:
        img_max = np.zeros_like(imgz)
    return(img_max)
def sum_z(imgz):
    try:
        img_sum = np.sum(imgz, axis=0)
    except:
        img_sum = np.zeros_like(imgz)
    return(img_sum)
def ave_z(imgz):
    try:
        img_ave = np.mean(imgz, axis=0)
    except:
        img_ave = np.zero_like(imgz)
    return(img_ave)

def simple_stitch(imgs, seq_num = 4, hr_margin = 233, vu_margin = 0):
    images = []
    for i in range(seq_num):
        img = imgs[i]
        if i < 3:
            images.append(img[:,:-hr_margin])
        else:
            images.append(img)
    img1 = np.hstack(images)
    images = []
    for i in range(seq_num):
        img = imgs[i+seq_num]
        if i < 3:
            images.append(img[:,:-hr_margin])
        else:
            images.append(img)
    img2 = np.hstack(images)
    im_h = np.vstack([img1[vu_margin:, :], img2[vu_margin:, :]])
    return(im_h)
def simple_stitch_z(imgs, z_num = None, seq_num = 4, hr_margin = 233, vu_margin = 0):
    if z_num == None:
        z_num = len(imgs[0])
    stitched = []
    for i in range(z_num):
        images = [img[i] for img in imgs]
        stitched.append(simple_stitch(images, seq_num, hr_margin, vu_margin))
    return(np.array(stitched))

def simple_stitch_alt(imgs, seq_num = 4, hr_margin = 233, vu_margin = 0):
    images = []
    for i in range(seq_num):
        img = imgs[i * 2]
        if i < 3:
            images.append(img[:,:-hr_margin])
        else:
            images.append(img)
    img1 = np.hstack(images)
    images = []
    for i in range(seq_num):
        img = imgs[i * 2 + 1]
        if i < 3:
            images.append(img[:,:-hr_margin])
        else:
            images.append(img)
    img2 = np.hstack(images)
    im_h = np.vstack([img1[vu_margin:, :], img2[vu_margin:, :]])
    return(im_h)
def simple_stitch_z_alt(imgs, z_num = None, seq_num = 4, hr_margin = 233, vu_margin = 0):
    if z_num == None:
        z_num = len(imgs[0])
    stitched = []
    for i in range(z_num):
        images = [img[i] for img in imgs]
        stitched.append(simple_stitch_alt(images, seq_num, hr_margin, vu_margin))
    return(np.array(stitched))

def sharping(img, cut_off = None):
    img_new = copy.deepcopy(img)
    if cut_off == None:
        cut_off = int(img.max() / 2)
    else:
        cut_off = int(img.max() * cut_off)
    img_new[img <= cut_off] = 0
    img_new[img > cut_off] = img.max()
    return(img_new)
def sum_y_int(img):
    img1d = np.sum(img, axis=0)
    return(img1d)
def slide_sum_y_int(img, slide_len = 100):
    img1d = np.sum(img, axis=0)
    img_slides = []
    for i in range(0, len(img1d) - slide_len):
        img_slides.append(np.sum(img1d[i:i+slide_len], axis=0))
    return(np.array(img_slides))
def sum_y_vol(img):
    img1d = np.count_nonzero(img, axis=0)
    return(img1d)
def slide_sum_y_vol(img, slide_len = 100):
    img1d = np.count_nonzero(img, axis=0)
    img_slides = []
    for i in range(0, len(img1d) - slide_len):
        img_slides.append(np.sum(img1d[i:i+slide_len], axis=0))
    return(np.array(img_slides))
def get_pop_density(img1d):
    pop_density = img1d/img1d.sum(axis=1, keepdims=True)
    return(pop_density)



def get_mask(img):
    img[img > 0] = 1
    img = np.abs(1 - img)
    return(img)
def get_mask_inv(img):
    img[img > 0] = 1
    return(img)
def mask_img(img_ori, mask):
    img = copy.deepcopy(img_ori)
    img[mask == 0] = 0
    return(img)
def simple_stitch_z_with_mask(imgs, mask, z_num = None, seq_num = 4, hr_margin = 233, vu_margin = 0):
    if z_num == None:
        z_num = len(imgs[0])
    stitched = []
    for i in range(z_num):
        images = [img[i] for img in imgs]
        stitched.append(mask_img(simple_stitch(images, seq_num, hr_margin, vu_margin), get_mask(mask)))
    return(np.array(stitched))


# %%
def shrink(img_ori, win=10):
    x,y = img_ori.shape
    xw = x//win
    yw = y//win
    new_img = []
    for i in range(xw):
        new_imgi = []
        for j in range(yw):
            new_imgi.append(np.sum(img_ori[i*win:i*win+10,j*win:j*win+10]))
        new_img.append(new_imgi)
    img = np.array(new_img).astype(np.uint16)
    return(img)


# %%
work_dir = "/Users/feng626/workspace/data/SoilSFA/spatial_interactions"
patch_number = 8
rolling_ball_sizes = [7, 9, 5]
min_thrs = [32, 48, 128]
sharping_thr = 0.01

rep1_dir = "/Users/feng626/workspace/data/SoilSFA/spatial_interactions/Activity_1.3_rep1_11.18.21_start"
rep3_dir = work_dir + "/Activity_1.3_rep3_02.07.22_start"
rep3_img_dirs = [name for name in os.listdir(rep3_dir) if os.path.isdir(os.path.join(rep3_dir, name))] 
rep3_img_dirs.sort()
rep2_dir = work_dir + "/Activity_1.3_rep2_12.13.21_start"
rep2_img_dirs = [name for name in os.listdir(rep2_dir) if os.path.isdir(os.path.join(rep2_dir, name))] 
rep2_img_dirs.sort()

# %%
# Replicate 1
result_dir = work_dir + "/results_dev/results_rep1"
for carbon in ["Chitin", "Chito5", "Nag"]:
# for carbon, timepoints in zip(["Chitin", "Chito5", "Nag"], [["11.22.21_17"], ["11.22.21_16"], ["11.22.21_15"]]):
    time_dir = rep1_dir + '/' + carbon 
    timepoints = [name for name in os.listdir(time_dir) if os.path.isdir(os.path.join(time_dir, name))] 
    timepoints.sort()
    for timepoint in timepoints:
        all_channels_ori = []
        all_channels = []
        all_channels_sharp = []
        for chi, channel in enumerate(['w4SD RFP', 'w3SD GFP', 'w2SD DAPI']):
            masked_file = rep1_dir + '/' + carbon + '/' + timepoint + '/' + timepoint + '_' + carbon + '_1_w1SD BF_stitched_MASK.tif'
            tif_files = [rep1_dir + '/' + carbon + '/' + timepoint + '/' + timepoint + '_' + carbon + '_1_' + channel + '_s' + str(i+1) + '.TIF' for i in range(patch_number)]
            tif_files.sort()
            imgm_tif_file= result_dir + '/' + timepoint + '_' + carbon + '_1_{}_masked.TIF'.format(channel)
            imgn_tif_file= result_dir + '/' + timepoint + '_' + carbon + '_1_{}_processed.TIF'.format(channel)
            imgn_sharp_tif_file= result_dir + '/' + timepoint + '_' + carbon + '_1_{}_sharp.TIF'.format(channel)
            projected_csv_file = result_dir + '/' + timepoint + '_' + carbon + '_1_{}_processed.csv'.format(channel)

            imgs_mask = get_mask(io.imread(masked_file))
            imgs = [io.imread(tif_file) for tif_file in tif_files]
            imgz = simple_stitch_z(imgs)
            imgp = max_z(imgz)
            imgp_rm_bg = remove_background(imgp, rolling_ball_sizes[chi])
            imgm = mask_img(imgp_rm_bg, imgs_mask)
            tifffile.imsave(imgm_tif_file, imgm)
            imgm_filtered = filter(imgm, min_thrs[chi])  # Don't use the minimal threshold
            imgn = normalize(imgm_filtered)
            tifffile.imsave(imgn_tif_file, imgn)
            np.savetxt(projected_csv_file, imgn, delimiter=',')
            imgn_sharp = sharping(imgn, sharping_thr)
            tifffile.imsave(imgn_sharp_tif_file, imgn_sharp)
    
            all_channels_ori.append(imgn)
            all_channels_sharp.append(imgn_sharp)
            
        combined_img_ori = np.transpose(all_channels_ori, (1,2,0))
        combined_img_sharp = np.transpose(all_channels_sharp, (1,2,0))
            
        output = result_dir + '/' + timepoint + '_' + carbon + '_1_combined_processed'
        img_out_ori = output + '.TIF'
        img_out_sharp = output + '_sharp.TIF'
        img_out_small_ori = output + '_small.TIF'
        img_out_small_sharp = output + '_small_sharp.TIF'
        pkl_out_ori = output + '.pkl'
        plot_out_ori = output + '_1d.png'
        plot_out_sharp = output + '_1d_sharp.png'
        
        tifffile.imsave(img_out_ori, combined_img_ori)
        tifffile.imsave(img_out_sharp, combined_img_sharp)
    
        with open(pkl_out_ori, 'wb') as f:
            pickle.dump(combined_img_ori, f)
            
        plt.figure(figsize=(16,8))
        plt.imshow(combined_img_ori)
        plt.savefig(img_out_small_ori, dpi=200)
        plt.close()
    
        plt.figure(figsize=(16,8))
        plt.imshow(combined_img_sharp)
        plt.savefig(img_out_small_sharp, dpi=200)
        plt.close()
    
        pop_den = get_pop_density(slide_sum_y_vol(combined_img_ori, 500))
        colors = ['red', 'green', 'blue']
        plt.figure(figsize=(16,8))
        for i in range(3):
            plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
        # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
        plt.savefig(plot_out_ori, dpi=200)
        # plt.show()
        plt.close()
    
        pop_den = get_pop_density(slide_sum_y_vol(combined_img_sharp, 500))
        colors = ['red', 'green', 'blue']
        plt.figure(figsize=(16,8))
        for i in range(3):
            plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
        # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
        plt.savefig(plot_out_sharp, dpi=200)
        # plt.show()
        plt.close()

# %%
# Replicate 2
result_dir = work_dir + "/results_dev/results_rep2"
for dir in rep2_img_dirs:
    img_dir = rep2_dir + '/' + dir 
    all_channels_ori = []
    all_channels = []
    all_channels_sharp = []
    for chi, channel in enumerate(['w4SD RFP', 'w3SD GFP', 'w2SD DAPI']):
        masked_file = img_dir + '/' + dir + '_1_w1SD BF_stitched_mask.tif'
        tif_files = [img_dir + '/' + dir + '_1_' + channel +  '_s' + str(i+1) + '.TIF' for i in range(patch_number)]
        tif_files.sort()
        imgm_tif_file= result_dir + '/' + dir + '_1_{}_masked.TIF'.format(channel)
        imgn_tif_file= result_dir + '/' + dir + '_1_{}_processed.TIF'.format(channel)
        imgn_sharp_tif_file= result_dir + '/' + dir + '_1_{}_sharp.TIF'.format(channel)
        projected_csv_file = result_dir + '/' + dir + '_1_{}_processed.csv'.format(channel)
        
        imgs_mask = get_mask(io.imread(masked_file))
        imgs = [io.imread(tif_file) for tif_file in tif_files]
        imgz = simple_stitch_z(imgs)
        imgp = max_z(imgz)
        imgp_rm_bg = remove_background(imgp, rolling_ball_sizes[chi])
        imgm = mask_img(imgp_rm_bg, imgs_mask)
        tifffile.imsave(imgm_tif_file, imgm)
        imgm_filtered = filter(imgm, min_thrs[chi])  # Don't use the minimal threshold
        imgn = normalize(imgm_filtered)
        tifffile.imsave(imgn_tif_file, imgn)
        np.savetxt(projected_csv_file, imgn, delimiter=',')
        imgn_sharp = sharping(imgn, sharping_thr)
        tifffile.imsave(imgn_sharp_tif_file, imgn_sharp)

        all_channels_ori.append(imgn)
        all_channels_sharp.append(imgn_sharp)
        
    combined_img_ori = np.transpose(all_channels_ori, (1,2,0))
    combined_img_sharp = np.transpose(all_channels_sharp, (1,2,0))
        
    output = result_dir + '/' + dir + '_1_combined_processed'
    img_out_ori = output + '.TIF'
    img_out_sharp = output + '_sharp.TIF'
    img_out_small_ori = output + '_small.TIF'
    img_out_small_sharp = output + '_small_sharp.TIF'
    pkl_out_ori = output + '.pkl'
    plot_out_ori = output + '_1d.png'
    plot_out_sharp = output + '_1d_sharp.png'
    
    tifffile.imsave(img_out_ori, combined_img_ori)
    tifffile.imsave(img_out_sharp, combined_img_sharp)

    with open(pkl_out_ori, 'wb') as f:
        pickle.dump(combined_img_ori, f)
        
    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_ori)
    plt.savefig(img_out_small_ori, dpi=200)
    plt.close()

    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_sharp)
    plt.savefig(img_out_small_sharp, dpi=200)
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_ori, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_ori, dpi=200)
    # plt.show()
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_sharp, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_sharp, dpi=200)
    # plt.show()
    plt.close()

# %%
# Replicate 3
result_dir = work_dir + "/results_dev/results_rep3"
for dir in rep3_img_dirs:
    # if dir == "02.09.22_11AM_Chito5":
    #     continue
    img_dir = rep3_dir + '/' + dir 
    all_channels_ori = []
    all_channels = []
    all_channels_sharp = []
    for chi, channel in enumerate(['w4SD RFP', 'w3SD GFP', 'w2SD DAPI']):
        masked_file = img_dir + '/' + dir + '_1_w1SD BF_stitched_mask.tif'
        tif_files = [img_dir + '/' + dir + '_1_' + channel +  '_s' + str(i+1) + '.TIF' for i in range(patch_number)]
        tif_files.sort()
        imgm_tif_file= result_dir + '/' + dir + '_1_{}_masked.TIF'.format(channel)
        imgn_tif_file= result_dir + '/' + dir + '_1_{}_processed.TIF'.format(channel)
        imgn_sharp_tif_file= result_dir + '/' + dir + '_1_{}_sharp.TIF'.format(channel)
        projected_csv_file = result_dir + '/' + dir + '_1_{}_processed.csv'.format(channel)
        
        imgs_mask = get_mask(io.imread(masked_file))
        imgs = [io.imread(tif_file) for tif_file in tif_files]
        imgz = simple_stitch_z(imgs)
        imgp = max_z(imgz)
        imgp_rm_bg = remove_background(imgp, rolling_ball_sizes[chi])
        imgm = mask_img(imgp_rm_bg, imgs_mask)
        tifffile.imsave(imgm_tif_file, imgm)
        imgm_filtered = filter(imgm, min_thrs[chi])  # Don't use the minimal threshold
        imgn = normalize(imgm_filtered)
        tifffile.imsave(imgn_tif_file, imgn)
        np.savetxt(projected_csv_file, imgn, delimiter=',')
        imgn_sharp = sharping(imgn, sharping_thr)
        tifffile.imsave(imgn_sharp_tif_file, imgn_sharp)

        all_channels_ori.append(imgn)
        all_channels_sharp.append(imgn_sharp)
        
    combined_img_ori = np.transpose(all_channels_ori, (1,2,0))
    combined_img_sharp = np.transpose(all_channels_sharp, (1,2,0))
        
    output = result_dir + '/' + dir + '_1_combined_processed'
    img_out_ori = output + '.TIF'
    img_out_sharp = output + '_sharp.TIF'
    img_out_small_ori = output + '_small.TIF'
    img_out_small_sharp = output + '_small_sharp.TIF'
    pkl_out_ori = output + '.pkl'
    plot_out_ori = output + '_1d.png'
    plot_out_sharp = output + '_1d_sharp.png'
    
    tifffile.imsave(img_out_ori, combined_img_ori)
    tifffile.imsave(img_out_sharp, combined_img_sharp)

    with open(pkl_out_ori, 'wb') as f:
        pickle.dump(combined_img_ori, f)
        
    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_ori)
    plt.savefig(img_out_small_ori, dpi=200)
    plt.close()

    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_sharp)
    plt.savefig(img_out_small_sharp, dpi=200)
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_ori, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_ori, dpi=200)
    # plt.show()
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_sharp, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_sharp, dpi=200)
    # plt.show()
    plt.close()

# %% [markdown]
# # Rep 7

# %%
work_dir = "/Users/feng626/workspace/data/SoilSFA/spatial_interactions"
results_dir = "/Users/feng626/Dropbox/PNNL/SoilSFA/image_analysis"
patch_number = 8
rolling_ball_sizes = [7, 9, 5]
min_thrs = [32, 48, 128]
sharping_thr = 0.01

# %%
rep7_dir = work_dir + "/Activity_1.3_rep7_10.11.22"
rep7_img_dirs = [name for name in os.listdir(rep7_dir) if os.path.isdir(os.path.join(rep7_dir, name))] 
rep7_img_dirs.sort()

# %%
# blank_patch = io.imread(rep7_dir + '/10.12.22_7PM_TP3_Chito5/10.12.22_7PM_TP3_Chito5_1_w2SD DAPI_s7.TIF')
# blank_patch = blank_patch * 0
# io.imsave(work_dir + '/rep7_blank_patch.TIF', blank_patch)

# %%
import shutil
mask_dir = work_dir + "/Activity_1.3_rep7_masks"
Path(mask_dir).mkdir(parents=True, exist_ok=True)
for dir in rep7_img_dirs:
    img_dir = rep7_dir + '/' + dir 
    all_channels_ori = []
    all_channels = []
    all_channels_sharp = []
    all_channels_zoomout = []
    masked_file = img_dir + '/' + dir + '_1_w1SD BF_stitched_mask.tif'
    shutil.copy(masked_file, mask_dir)

# %%
mask_dir = results_dir + "/results/Activity_1.3_rep7_masks"
mask_zoomout_dir = results_dir + "/results/Activity_1.3_rep7_masks_zoomout"
mask_files = os.listdir(mask_dir)
for mask_file in mask_files:
    mask = io.imread(mask_dir + '/' + mask_file)
    mask_small = shrink(mask, win=10)
    mask_small[mask_small > 0] = 255
    mask_small = mask_small.astype("uint8")
    zoomout_file = mask_zoomout_dir + '/' + mask_file
    io.imsave(zoomout_file, mask_small)

    mask_zoomout = copy.deepcopy(mask_small)
    mask_zoomout[mask_zoomout > 0] = 1
    mask_zoomout = np.abs(1 - mask_zoomout)
    zoomout_csv = mask_zoomout_dir + '/' + mask_file + ".csv"
    np.savetxt(zoomout_csv, mask_zoomout, delimiter=',')

# %%
print(mask_zoomout.dtype)

# %%
# Replicate 7
result_dir = results_dir + "/results/20230224/Activity_1.3_rep7"
Path(result_dir).mkdir(parents=True, exist_ok=True)
for dir in rep7_img_dirs:
    # if dir == "10.12.22_7PM_TP3_Chito5":
    #     print("This batch has patch 8 missing, may need to check in future!")
    img_dir = rep7_dir + '/' + dir 
    all_channels_ori = []
    all_channels = []
    all_channels_sharp = []
    all_channels_zoomout = []
    for chi, channel in enumerate(['w4SD RFP', 'w3SD GFP', 'w2SD DAPI']):
        masked_file = img_dir + '/' + dir + '_1_w1SD BF_stitched_mask.tif'
        tif_files = [img_dir + '/' + dir + '_1_' + channel +  '_s' + str(i+1) + '.TIF' for i in range(patch_number)]
        tif_files.sort()
        imgm_tif_file= result_dir + '/' + dir + '_1_{}_masked.TIF'.format(channel)
        imgn_tif_file= result_dir + '/' + dir + '_1_{}_processed.TIF'.format(channel)
        imgn_sharp_tif_file= result_dir + '/' + dir + '_1_{}_sharp.TIF'.format(channel)
        # projected_csv_file = result_dir + '/' + dir + '_1_{}_processed.csv'.format(channel)
        projected_shrink_csv_file = result_dir + '/' + dir + '_1_{}_processed_shrink.csv'.format(channel)
        
        imgs_mask = get_mask(io.imread(masked_file))
        imgs = [io.imread(tif_file) for tif_file in tif_files]
        imgz = simple_stitch_z(imgs)
        imgp = max_z(imgz)
        imgp_rm_bg = remove_background(imgp, rolling_ball_sizes[chi])
        imgm = mask_img(imgp_rm_bg, imgs_mask)
        # tifffile.imsave(imgm_tif_file, imgm)
        imgm_filtered = filter(imgm, min_thrs[chi])  # Don't use the minimal threshold
        # imgm_filtered = filter(imgm)  # Don't use the minimal threshold
        imgn = normalize(imgm_filtered)
        # tifffile.imsave(imgn_tif_file, imgn)
        # np.savetxt(projected_csv_file, imgn, delimiter=',')
        imgn_small = shrink(imgn, win=10)
        np.savetxt(projected_shrink_csv_file, imgn_small, delimiter=',')
        imgn_sharp = sharping(imgn, sharping_thr)
        # tifffile.imsave(imgn_sharp_tif_file, imgn_sharp)

        all_channels_ori.append(imgn)
        all_channels_sharp.append(imgn_sharp)
        all_channels_zoomout.append(imgn_small)
        
    combined_img_ori = np.transpose(all_channels_ori, (1,2,0))
    combined_img_sharp = np.transpose(all_channels_sharp, (1,2,0))
    combined_img_zoomout = np.transpose(all_channels_zoomout, (1,2,0))
        
    output = result_dir + '/' + dir + '_1_combined_processed'
    img_out_ori = output + '.TIF'
    img_out_sharp = output + '_sharp.TIF'
    img_out_zoomout = output + '_zoomout.TIF'
    img_out_small_ori = output + '_small.TIF'
    img_out_small_sharp = output + '_sharp_small.TIF'
    img_out_small_zoomout = output + '_zoomout_small.TIF'
    pkl_out_ori = output + '.pkl'
    plot_out_ori = output + '_1d.png'
    plot_out_sharp = output + '_1d_sharp.png'
    
    tifffile.imsave(img_out_ori, combined_img_ori)
    tifffile.imsave(img_out_sharp, combined_img_sharp)
    tifffile.imsave(img_out_zoomout, combined_img_zoomout)

    with open(pkl_out_ori, 'wb') as f:
        pickle.dump(combined_img_ori, f)
        
    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_ori)
    plt.savefig(img_out_small_ori, dpi=200)
    plt.close()

    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_sharp)
    plt.savefig(img_out_small_sharp, dpi=200)
    plt.close()

    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_zoomout)
    plt.savefig(img_out_small_zoomout, dpi=200)
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_ori, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_ori, dpi=200)
    # plt.show()
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_sharp, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_sharp, dpi=200)
    # plt.show()
    plt.close()

# %%
min_thrs = [32, 48, 256]
# Replicate 7
result_dir = results_dir + "/results/20230224/Activity_1.3_rep7"
Path(result_dir).mkdir(parents=True, exist_ok=True)
for dir in rep7_img_dirs:
    # if dir == "10.12.22_7PM_TP3_Chito5":
    #     print("This batch has patch 8 missing, may need to check in future!")
    img_dir = rep7_dir + '/' + dir 
    all_channels_ori = []
    all_channels = []
    all_channels_sharp = []
    all_channels_zoomout = []
    for chi, channel in enumerate(['w4SD RFP', 'w3SD GFP', 'w2SD DAPI']):
        masked_file = img_dir + '/' + dir + '_1_w1SD BF_stitched_mask.tif'
        tif_files = [img_dir + '/' + dir + '_1_' + channel +  '_s' + str(i+1) + '.TIF' for i in range(patch_number)]
        tif_files.sort()
        imgm_tif_file= result_dir + '/' + dir + '_1_{}_masked.TIF'.format(channel)
        imgn_tif_file= result_dir + '/' + dir + '_1_{}_processed.TIF'.format(channel)
        imgn_sharp_tif_file= result_dir + '/' + dir + '_1_{}_sharp.TIF'.format(channel)
        # projected_csv_file = result_dir + '/' + dir + '_1_{}_processed.csv'.format(channel)
        projected_shrink_csv_file = result_dir + '/' + dir + '_1_{}_processed_shrink.csv'.format(channel)
        
        imgs_mask = get_mask(io.imread(masked_file))
        imgs = [io.imread(tif_file) for tif_file in tif_files]
        imgz = simple_stitch_z(imgs)
        imgp = max_z(imgz)
        imgp_rm_bg = remove_background(imgp, rolling_ball_sizes[chi])
        imgm = mask_img(imgp_rm_bg, imgs_mask)
        # tifffile.imsave(imgm_tif_file, imgm)
        imgm_filtered = filter(imgm, min_thrs[chi])  # Don't use the minimal threshold
        # imgm_filtered = filter(imgm)  # Don't use the minimal threshold
        imgn = normalize(imgm_filtered)
        # tifffile.imsave(imgn_tif_file, imgn)
        # np.savetxt(projected_csv_file, imgn, delimiter=',')
        imgn_small = shrink(imgn, win=10)
        np.savetxt(projected_shrink_csv_file, imgn_small, delimiter=',')
        imgn_sharp = sharping(imgn, sharping_thr)
        # tifffile.imsave(imgn_sharp_tif_file, imgn_sharp)

        all_channels_ori.append(imgn)
        all_channels_sharp.append(imgn_sharp)
        all_channels_zoomout.append(imgn_small)
        
    combined_img_ori = np.transpose(all_channels_ori, (1,2,0))
    combined_img_sharp = np.transpose(all_channels_sharp, (1,2,0))
    combined_img_zoomout = np.transpose(all_channels_zoomout, (1,2,0))
        
    output = result_dir + '/' + dir + '_1_combined_processed'
    img_out_ori = output + '.TIF'
    img_out_sharp = output + '_sharp.TIF'
    img_out_zoomout = output + '_zoomout.TIF'
    img_out_small_ori = output + '_small.TIF'
    img_out_small_sharp = output + '_small_sharp.TIF'
    img_out_small_zoomout = output + '_zoomout_small.TIF'
    pkl_out_ori = output + '.pkl'
    plot_out_ori = output + '_1d.png'
    plot_out_sharp = output + '_1d_sharp.png'
    
    tifffile.imsave(img_out_ori, combined_img_ori)
    tifffile.imsave(img_out_sharp, combined_img_sharp)
    tifffile.imsave(img_out_zoomout, combined_img_zoomout)

    with open(pkl_out_ori, 'wb') as f:
        pickle.dump(combined_img_ori, f)
        
    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_ori)
    plt.savefig(img_out_small_ori, dpi=200)
    plt.close()

    plt.figure(figsize=(16,8))
    plt.imshow(combined_img_sharp)
    plt.savefig(img_out_small_sharp, dpi=200)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.imshow(combined_img_zoomout)
    plt.savefig(img_out_small_zoomout, dpi=100)
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_ori, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_ori, dpi=200)
    # plt.show()
    plt.close()

    pop_den = get_pop_density(slide_sum_y_vol(combined_img_sharp, 500))
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(16,8))
    for i in range(3):
        plt.plot(range(len(pop_den)), pop_den[:,i], color = colors[i], linewidth=3)
    # plt.plot(range(len(pop_den)), pop_den, linewidth=2)
    plt.savefig(plot_out_sharp, dpi=200)
    # plt.show()
    plt.close()
