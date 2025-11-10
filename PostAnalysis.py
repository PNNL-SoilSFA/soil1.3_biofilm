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
# title: Temporal dynamics
# jupyter: python3
# ---

# %%
import warnings, time, datetime
warnings.filterwarnings("ignore")
import math, os, sys, copy, pickle
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd
# Make images higher resolution and set default size
import matplotlib
# %matplotlib inline
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (5, 5)
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# %%
from skimage import io, filters, restoration
from skimage import morphology as morph
import PIL
from PIL import Image, ImageSequence, ImageEnhance
import glob
import cv2
import multipagetiff as mtif
import tifffile
import plotly.express as px
import plotly.graph_objs as go
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%
def get_pop_density(img1d):
    pop_density = img1d/img1d.sum(axis=1, keepdims=True)
    return(pop_density)
def slide_sum_y_vol(img, slide_len = 100):
    img1d = np.count_nonzero(img, axis=0)
    img_slides = []
    for i in range(0, len(img1d) - slide_len):
        img_slides.append(np.sum(img1d[i:i+slide_len], axis=0))
    return(np.array(img_slides))


# %%
work_dir = "/Users/feng626/workspace/data/soil1.3/spatial_interactions/"
result_dir = work_dir + "20220408/"

# analysis_dir = "./results/{}/".format(datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S.%f"))
# analysis_dir = "./results/{}/".format(datetime.datetime.now().strftime("%Y%m%d"))
analysis_dir = "./results/20240226/"
Path(analysis_dir).mkdir(parents=True, exist_ok=True)

# %%
pixel_size = 0.125
pixel_area = 0.125 ** 2

# %%
rep1_samples = [['11.18.21_21', 'Nag', 0],
                ['11.19.21_08', 'Nag', 11],
                # ['11.19.21_18', 'Nag', 21],
                ['11.19.21_18', 'Nag', 22],
                ['11.20.21_10', 'Nag', 37],
                ['11.21.21_15', 'Nag', 66],
                ['11.22.21_15', 'Nag', 90],
                ['11.18.21_22', 'Chito5', 0],
                ['11.19.21_09', 'Chito5', 11],
                # ['11.19.21_19', 'Chito5', 21],
                ['11.19.21_19', 'Chito5', 22],
                ['11.20.21_11', 'Chito5', 37],
                ['11.21.21_16', 'Chito5', 66],
                ['11.22.21_16', 'Chito5', 90],
                ['11.18.21_23', 'Chitin', 0],
                ['11.19.21_10', 'Chitin', 11],
                # ['11.19.21_20', 'Chitin', 21],
                ['11.19.21_20', 'Chitin', 22],
                ['11.20.21_12', 'Chitin', 37],
                ['11.21.21_17', 'Chitin', 66],
                ['11.22.21_17', 'Chitin', 90]]

# %%
rep2_samples = [["12.13.21_9PM", "Nag", 0],
                ["12.14.21_8AM", "Nag", 11],
                ["12.14.21_7PM", "Nag", 22],
                # ["12.15.21_8AM", "Nag", 35],
                ["12.15.21_8AM", "Nag", 37],
                ["12.16.21_3PM", "Nag", 66],
                ["12.17.21_3PM", "Nag", 90],
                ["12.13.21_10PM", "Chito5", 0],
                ["12.14.21_9AM", "Chito5", 11],
                ["12.14.21_8PM", "Chito5", 22],
                # ["12.15.21_9AM", "Chito5", 35],
                ["12.15.21_9AM", "Chito5", 37],
                ["12.16.21_4PM", "Chito5", 66],
                ["12.17.21_4PM", "Chito5", 90],
                ["12.13.21_11PM", "Chitin", 0],
                ["12.14.21_10AM", "Chitin", 11],
                ["12.14.21_9PM", "Chitin", 22],
                # ["12.15.21_10AM", "Chitin", 35],
                ["12.15.21_10AM", "Chitin", 37],
                ["12.16.21_5PM", "Chitin", 66],
                ["12.17.21_5PM", "Chitin", 90]]

# %%
rep3_samples = [["02.07.22_9PM", "Nag", 0],
                ["02.08.22_8AM", "Nag", 11],
                ["02.08.22_7PM", "Nag", 22],
                ["02.09.22_10AM", "Nag", 37],
                ["02.10.22_3PM", "Nag", 66],
                ["02.11.22_3PM", "Nag", 90],
                ["02.07.22_10PM", "Chito5", 0],
                ["02.08.22_9AM", "Chito5", 11],
                ["02.08.22_8PM", "Chito5", 22],
                ["02.09.22_11AM", "Chito5", 37],
                ["02.10.22_4PM", "Chito5", 66],
                ["02.11.22_4PM", "Chito5", 90],
                ["02.07.22_11PM", "Chitin", 0],
                ["02.08.22_10AM", "Chitin", 11],
                ["02.08.22_9PM", "Chitin", 22],
                ["02.09.22_12PM", "Chitin", 37],
                ["02.10.22_5PM", "Chitin", 66],
                ["02.11.22_5PM", "Chitin", 90]]

# %%
pop_densities = []
pop_volumes = []
for rep_samples, folder in zip([rep1_samples, rep2_samples, rep3_samples], ["Activity_1.3_rep" + str(i+1) for i in range(3)]):
    mask_dir = work_dir + folder + "_masks/"
    rep_dir = result_dir + folder    
    for sample in rep_samples:
        mask_file = mask_dir + sample[0] + '_' + sample[1] + "_1_w1SD BF_stitched_mask.tif"
        mask_array = io.imread(mask_file)
        mask_array[mask_array > 0] = 1
        mask_array = np.abs(mask_array - 1)
        rgb_file = rep_dir + '/' + sample[0] + '_' + sample[1] + "_1_combined_processed.pkl"
        with open(rgb_file, 'rb') as f:
            rgb_array = pickle.load(f)
        mask_vol = mask_array.sum() * pixel_area
        pops = rgb_array.sum(axis=(0,1)) * pixel_area
        if sample[1] == "Nag":
            # sample_carbon_name = "N-acetylglucosamine"
            sample_carbon_name = "NAG"
        elif sample[1] == "Chito5":
            sample_carbon_name = "Chitopentose"
        elif sample[1] == "Chitin":
            sample_carbon_name = "Chitin"
        else:
            sample_carbon_name = "Unknown"
        pop_volumes.append([folder.split('_')[2], sample_carbon_name, sample[2], pops[0], pops[1], pops[2], mask_vol])
        pop_den = pops / pops.sum()
        pop_densities.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_den[0], pop_den[1], pop_den[2], mask_vol])
# pop_den_table = [list(pop_deni) for pop_deni in zip(*pop_densities)]

pop_den_df = pd.DataFrame(pop_densities, columns = ["Rep", "Nutrient", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Volume"])
pop_vol_df = pd.DataFrame(pop_volumes, columns = ["Rep", "Nutrient", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Volume"])
pop_vol_df["Total"] = pop_vol_df["Variovorax"] + pop_vol_df["Sphingopyxis"] + pop_vol_df["Rhodococcus"]
pop_vol_df["Variovorax / Volume"] = pop_vol_df["Variovorax"] / pop_vol_df["Volume"]
pop_vol_df["Sphingopyxis / Volume"] = pop_vol_df["Sphingopyxis"] / pop_vol_df["Volume"]
pop_vol_df["Rhodococcus/ Volume"] = pop_vol_df["Rhodococcus"] / pop_vol_df["Volume"]

pop_den_df.to_pickle(analysis_dir + "pop_densities.pkl")
pop_vol_df.to_pickle(analysis_dir + "pop_volumes.pkl")

# %%
pop_den_df = pd.read_pickle(analysis_dir + "pop_densities.pkl")
pop_vol_df = pd.read_pickle(analysis_dir + "pop_volumes.pkl")

# %%
# pop_vol_df["Sample"] = pop_vol_df["Nutrient"] + "_" + pop_vol_df["Time"].astype(str) + "_" + pop_vol_df["Rep"]
pop_vol_df["Sample"] = pop_vol_df["Nutrient"] + "_" + pop_vol_df["Rep"]
pop_vol_df.head()

# %%
from dfply import *
import pingouin as pg

# %%
mixed_anova2 = pg.mixed_anova(dv='Total', between='Nutrient', within='Time', subject='Sample', data=pop_vol_df)
# print(mixed_anova2)
mixed_anova2.to_csv(analysis_dir + "mixed_anova2.csv")
mixed_anova2

# %%
df_anova = pd.melt(pop_vol_df, id_vars=["Sample", "Nutrient", "Time", "Rep", "Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")
df_anova.head()

# %%
anova3 = df_anova.anova(dv='Biomass', between=['Nutrient', 'Strain', 'Time'], ss_type=3)#.round(5)
anova3.to_csv(analysis_dir + "anova3.csv")
anova3

# %%
anova21 = df_anova.anova(dv='Biomass', between=['Nutrient', 'Strain'], ss_type=3)#.round(5)
anova21.to_csv(analysis_dir + "anova21.csv")
anova21

# %%
anova22 = df_anova.anova(dv='Biomass', between=['Nutrient', 'Time'], ss_type=3)#.round(5)
anova22.to_csv(analysis_dir + "anova22.csv")
anova22

# %%
anova23 = df_anova.anova(dv='Biomass', between=['Strain', 'Time'], ss_type=3)#.round(5)
anova23.to_csv(analysis_dir + "anova23.csv")
anova23

# %%
ancova_nutrient = df_anova.ancova(dv='Biomass', covar=["Time"], between='Nutrient')#.round(5)
ancova_nutrient.to_csv(analysis_dir + "ancova_nutrient.csv")
ancova_nutrient

# %%
ancova_strain = df_anova.ancova(dv='Biomass', covar=["Time"], between='Strain')#.round(5)
ancova_strain.to_csv(analysis_dir + "ancova_strain.csv")
ancova_strain

# %%
df = pop_den_df.copy()
df["Time"] = (df["Time"]) / 5 + 2
fig = px.scatter_ternary(df[pop_den_df["Rep"] == "rep1"], 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep1.pdf")

# %%
df = pop_den_df.copy()
df["Time"] = (df["Time"]) / 5 + 2
fig = px.scatter_ternary(df[pop_den_df["Rep"] == "rep2"],
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep2.pdf")

# %%
df = pop_den_df.copy()
df["Time"] = (df["Time"]) / 5 + 2
fig = px.scatter_ternary(df[pop_den_df["Rep"] == "rep3"], 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep3.pdf")

# %%
df = pop_den_df.copy()
df["Time"] = (df["Time"]) / 5 + 2
fig = px.scatter_ternary(df, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient", symbol="Rep")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep123.pdf")

# %%
df = pop_den_df.copy()
df["Time"] = (df["Time"]) / 5 + 2
fig = px.scatter_ternary(df, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.update_layout(legend= {'itemsizing': 'constant'})
fig.show()
fig.write_image(analysis_dir + "rep123_all_reps.pdf")

# %%
df = pop_den_df.copy()
df["Time"] = df["Time"].astype(str) + " h"
fig = px.scatter_ternary(df, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         color="Time", symbol="Nutrient")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep123_all_reps_alt.pdf")

# %%
df = pop_den_df.copy()
# df["Time"] = df["Time"].astype(str) + " h"
fig = px.scatter_ternary(df, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         color="Time", symbol="Nutrient", color_continuous_scale=px.colors.sequential.turbid)
fig.update_traces(marker={'size': 15})
fig.update_coloraxes(colorbar_x=1.03, colorbar_y=0.52)
fig.update_layout(legend_x=0.8, legend_y=1.0)
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep123_all_reps_alt_cont.pdf")

# %%
df = pop_den_df.copy()
# df["Time"] = df["Time"].astype(str) + " h"
df["Time"] = (df["Time"]) / 5 + 2
fig = px.scatter_ternary(df, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient")
# fig.update_coloraxes(colorbar_x=1.0, colorbar_y=0.52)
fig.update_layout(legend= {'itemsizing': 'trace'})
# fig.update_layout(legend_x=0.8, legend_y=1.0)
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep123_all_reps_alt_cont_size.pdf")

# %% [markdown]
# ## Spatiotemporal dynamics
#
#
# ## Spatial dynamics

# %%
slide_win = 500
mask_volumes_1d = []
pop_volumes_1d = []
delta_pop_volumes_1d = []
mask_density_1d = []
pop_densities_1d = []
delta_pop_densities_1d = []
for rep_samples, folder in zip([rep1_samples, rep2_samples, rep3_samples], ["Activity_1.3_rep" + str(i+1) for i in range(3)]):
    mask_dir = work_dir + folder + "_masks/"
    rep_dir = result_dir + folder + '/'    
    for sample in rep_samples:
        mask_file = mask_dir + sample[0] + '_' + sample[1] + "_1_w1SD BF_stitched_mask.tif"
        mask_array = io.imread(mask_file)
        mask_array[mask_array > 0] = 1
        mask_array = np.abs(mask_array - 1)
        rgb_file = rep_dir + sample[0] + '_' + sample[1] + "_1_combined_processed.pkl"
        with open(rgb_file, 'rb') as f:
            rgb_array = pickle.load(f)
        mask_vol = slide_sum_y_vol(mask_array, slide_win) * pixel_area
        pop_vol = slide_sum_y_vol(rgb_array, slide_win) * pixel_area
        if sample[1] == "Nag":
            # sample_carbon_name = "N-acetylglucosamine"
            sample_carbon_name = "NAG"
        elif sample[1] == "Chito5":
            sample_carbon_name = "Chitopentose"
        elif sample[1] == "Chitin":
            sample_carbon_name = "Chitin"
        else:
            sample_carbon_name = "Unknown"
        for i in range(len(pop_vol)):
            pop_volumes_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_vol[i,0], pop_vol[i,1], pop_vol[i,2], i * pixel_size, mask_vol[i]])
            if i > 0:
                delta_pop_volumes_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_vol[i,0] - pop_vol[i-1,0], pop_vol[i,1] - pop_vol[i-1,1], pop_vol[i,2] - pop_vol[i-1,2], i * pixel_size, 1, mask_vol[i] - mask_vol[i-1]])
            if i >= 10:
                delta_pop_volumes_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_vol[i,0] - pop_vol[i-10,0], pop_vol[i,1] - pop_vol[i-10,1], pop_vol[i,2] - pop_vol[i-10,2], i * pixel_size, 10, mask_vol[i] - mask_vol[i-10]])
            if i >= 100:
                delta_pop_volumes_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_vol[i,0] - pop_vol[i-100,0], pop_vol[i,1] - pop_vol[i-100,1], pop_vol[i,2] - pop_vol[i-100,2], i * pixel_size, 100, mask_vol[i] - mask_vol[i-100]])
        pop_den = get_pop_density(slide_sum_y_vol(rgb_array, slide_win))
        for i in range(len(pop_den)):
            pop_densities_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_den[i,0], pop_den[i,1], pop_den[i,2], i * pixel_size, mask_vol[i]])
            if i > 0:
                delta_pop_densities_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_den[i,0] - pop_den[i-1,0], pop_den[i,1] - pop_den[i-1,1], pop_den[i,2] - pop_den[i-1,2], i * pixel_size, 1, mask_vol[i] - mask_vol[i-1]])
            if i >= 10:
                delta_pop_densities_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_den[i,0] - pop_den[i-10,0], pop_den[i,1] - pop_den[i-10,1], pop_den[i,2] - pop_den[i-10,2], i * pixel_size, 10, mask_vol[i] - mask_vol[i-10]])
            if i >= 100:
                delta_pop_densities_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_den[i,0] - pop_den[i-100,0], pop_den[i,1] - pop_den[i-100,1], pop_den[i,2] - pop_den[i-100,2], i * pixel_size, 100, mask_vol[i] - mask_vol[i-100]])
# pop_den_table = [list(pop_deni) for pop_deni in zip(*pop_densities)]

# %%
spatial_dir = analysis_dir + "spatial/"
Path(spatial_dir).mkdir(parents=True, exist_ok=True)

# %%
pop_vol_1d_df = pd.DataFrame(pop_volumes_1d, columns = ["Rep", "Nutrient", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Radius", "Volume"])
pop_vol_1d_df["Relative Volume"] = pop_vol_1d_df["Volume"] / pop_vol_1d_df["Volume"].max()
pop_vol_1d_df["Variovorax / Volume"] = pop_vol_1d_df["Variovorax"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df["Sphingopyxis / Volume"] = pop_vol_1d_df["Sphingopyxis"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df["Rhodococcus/ Volume"] = pop_vol_1d_df["Rhodococcus"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df["Total"] = pop_vol_1d_df["Variovorax"] + pop_vol_1d_df["Sphingopyxis"] + pop_vol_1d_df["Rhodococcus"]
pop_vol_1d_df["Total / Volume"] = pop_vol_1d_df["Total"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df.to_pickle(analysis_dir + "spatial/pop_vol_1d_df_" + str(slide_win) + ".pkl")
delta_pop_vol_1d_df = pd.DataFrame(delta_pop_volumes_1d, columns = ["Rep", "Nutrient", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Radius", "Step", "Volume"])
delta_pop_vol_1d_df.to_pickle(analysis_dir + "spatial/delta_pop_vol_1d_df_" + str(slide_win) + ".pkl")


pop_den_1d_df = pd.DataFrame(pop_densities_1d, columns = ["Rep", "Nutrient", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Radius", "Volume"])
pop_den_1d_df["Relative Volume"] = pop_den_1d_df["Volume"] / pop_den_1d_df["Volume"].max()
# pop_den_1d_df["Variovorax / Volume"] = pop_den_1d_df["Variovorax"] / pop_den_1d_df["Volume"]
# pop_den_1d_df["Sphingopyxis / Volume"] = pop_den_1d_df["Sphingopyxis"] / pop_den_1d_df["Volume"]
# pop_den_1d_df["Rhodococcus/ Volume"] = pop_den_1d_df["Rhodococcus"] / pop_den_1d_df["Volume"]
pop_den_1d_df.to_pickle(analysis_dir + "spatial/pop_den_1d_df_" + str(slide_win) + ".pkl")
delta_pop_den_1d_df = pd.DataFrame(delta_pop_densities_1d, columns = ["Rep", "Nutrient", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Radius", "Step", "Volume"])
delta_pop_den_1d_df.to_pickle(analysis_dir + "spatial/delta_pop_den_1d_df_" + str(slide_win) + ".pkl")

# %%
slide_win = 500
pop_vol_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_vol_1d_df_" + str(slide_win) + ".pkl")
delta_pop_vol_1d_df = pd.read_pickle(analysis_dir + "spatial/delta_pop_vol_1d_df_" + str(slide_win) + ".pkl")
pop_den_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_den_1d_df_" + str(slide_win) + ".pkl")
delta_pop_den_1d_df = pd.read_pickle(analysis_dir + "spatial/delta_pop_den_1d_df_" + str(slide_win) + ".pkl")

# %%
df = pop_den_1d_df.copy()
dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Rep"] == "rep3")].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66)].copy()
fig = px.line(dft, x='Radius', y=['Variovorax', 'Sphingopyxis', 'Rhodococcus', 'Relative Volume'], title='1D with Nag at 66h')
# fig.write_image(analysis_dir + "spatial/1D_with_Nag_at_66h_" + str(slide_win) + ".pdf")
fig.show()

# %%
df = pop_den_1d_df.copy()
dft = df[(df["Nutrient"] == "Chitopentose") & (df["Time"] == 66) & (df["Rep"] == "rep3")].copy()
# dft = df[(df["Nutrient"] == "Chitopentose") & (df["Time"] == 66)].copy()
fig = px.line(dft, x='Radius', y=['Variovorax', 'Sphingopyxis', 'Rhodococcus', 'Relative Volume'], title='1D with Chito5 at 66h')
# fig.write_image(analysis_dir + "spatial/1D_with_Chito5_at_66h_" + str(slide_win) + ".pdf")
fig.show()

# %%
df = pop_den_1d_df.copy()
dft = df[(df["Nutrient"] == "Chitin") & (df["Time"] == 66) & (df["Rep"] == "rep3")].copy()
# dft = df[(df["Nutrient"] == "Chitin") & (df["Time"] == 66)].copy()
fig = px.line(dft, x='Radius', y=['Variovorax', 'Sphingopyxis', 'Rhodococcus', 'Relative Volume'], title='1D with Chitin at 66h')
# fig.write_image(analysis_dir + "spatial/1D_with_Chitin_at_66h_" + str(slide_win) + ".pdf")
fig.show()

# %%
# df = pop_den_1d_df.copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Rep"] == "rep3")].copy()
# # dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66)].copy()
# fig = px.line_3d(dft, x='Variovorax', y='Sphingopyxis', z='Rhodococcus', title='1D with at 66h')
# fig.show()

# %%
df = pop_den_1d_df.copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Rep"] == "rep3") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Rep"] == "rep3")].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "Chitopentose") & (df["Time"] == 66) & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "Chitin") & (df["Time"] == 66) & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66)].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "Chitopentose") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
dft = df[(df["Nutrient"] == "Chitin") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "NAG")].copy()
dft = df[(df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df.copy()
fig = px.scatter_ternary(dft,
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         color="Relative Volume", color_continuous_scale="Viridis")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()

# %%
df = pop_den_1d_df.copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Rep"] == "rep3") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Rep"] == "rep3")].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "Chitopentose") & (df["Time"] == 66) & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "Chitin") & (df["Time"] == 66) & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66)].copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "Chitopentose") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "Chitin") & (df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df[(df["Nutrient"] == "NAG")].copy()
dft = df[(df["Radius"] % (100 * pixel_size) == 0)].copy()
# dft = df.copy()
fig = px.scatter_ternary(dft,
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         color="Radius")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()

# %%
# df = delta_pop_den_1d_df.copy()
# dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66) & (df["Rep"] == "rep3")
#          & (df["Step"] == 10)
#          ].copy()
# # dft = df[(df["Nutrient"] == "NAG") & (df["Time"] == 66)].copy()
# fig = px.scatter_3d(dft.loc[(df[['Variovorax', 'Sphingopyxis', 'Rhodococcus']] != 0).all(axis=1)], x='Variovorax', y='Sphingopyxis', z='Rhodococcus', title='1D with at 66h')
# fig.update_traces(
#     marker=dict(size=2)
# )
# fig.show()

# %%
fig = px.scatter(dft.loc[(df[['Variovorax', 'Sphingopyxis', 'Rhodococcus']] != 0).all(axis=1)], x='Variovorax', y='Sphingopyxis', title='1D with at 66h')
fig.update_traces(
    marker=dict(size=2)
)
fig.show()

# %%
fig = px.scatter(dft.loc[(df[['Variovorax', 'Sphingopyxis', 'Rhodococcus']] != 0).all(axis=1)], x='Rhodococcus', y='Sphingopyxis', title='1D with at 66h')
fig.update_traces(
    marker=dict(size=2)
)
fig.show()

# %%
df4plot = pop_den_1d_df[(pop_den_1d_df["Rep"] == "rep1") & ((pop_den_1d_df["Radius"] == (0 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == ((750 + slide_win//2) * pixel_size)) | (pop_den_1d_df["Radius"] == (1500 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (2250 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (3000 + slide_win//2) * pixel_size))]
df4plot["Time"] = (df4plot["Time"]) / 5 + 2
fig = px.scatter_ternary(df4plot,
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient", symbol="Radius")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "spatial/1Drep1_" + str(slide_win) + ".pdf")

# %%
df4plot = pop_den_1d_df[(pop_den_1d_df["Rep"] == "rep2") & ((pop_den_1d_df["Radius"] == (0 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (750 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (1500 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (2250 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (3000 + slide_win//2) * pixel_size))]
df4plot["Time"] = (df4plot["Time"]) / 5 + 2
fig = px.scatter_ternary(df4plot, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient", symbol="Radius")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "spatial/1Drep2_" + str(slide_win) + ".pdf")

# %%
df4plot = pop_den_1d_df[(pop_den_1d_df["Rep"] == "rep3") & ((pop_den_1d_df["Radius"] == (0 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (750 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (1500 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (2250 + slide_win//2) * pixel_size) | (pop_den_1d_df["Radius"] == (3000 + slide_win//2) * pixel_size))]
df4plot["Time"] = (df4plot["Time"]) / 5 + 2
fig = px.scatter_ternary(df4plot, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         size="Time", color="Nutrient", symbol="Radius")
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "spatial/1Drep3_" + str(slide_win) + ".pdf")

# %%
times = [0, 11, 22, 37, 66, 90]
exploratory_dir = result_dir + "spatial/exploratory/"
Path(exploratory_dir).mkdir(parents=True, exist_ok=True)

# %%
df = pop_den_1d_df.copy()
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Relative Volume': 'grey'}
for time in times:
    for rep in ["rep" + str(i+1) for i in range(3)]:
        dfr = df[(df["Time"] == time) & (df["Rep"] == rep)].copy()
        # ymax = dfr[['Variovorax', 'Sphingopyxis', 'Rhodococcus']].values.flatten().max()
        for carbon in ["NAG", "Chitopentose", "Chitin"]:
            dft = dfr[(dfr["Nutrient"] == carbon)].copy()
            fig = px.line(dft, x='Radius', y=['Variovorax', 'Sphingopyxis', 'Rhodococcus', 'Relative Volume'], title="Population density of " + rep + " with " + carbon + " at " + str(time) + "h", color_discrete_map=color_map, template="ggplot2")
            # fig.show()
            fig.write_image(analysis_dir + "spatial/exploratory/pop_den_" + rep +"_with_" + carbon + "_at_" + str(time) + "h_" + str(slide_win) + ".pdf", scale=1, width=1000, height=500)

# %%
df = pop_vol_1d_df.copy()
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Volume': 'grey'}
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
for carbon in ["NAG", "Chitopentose", "Chitin"]:
    for rep in ["rep" + str(i+1) for i in range(3)]:
        dfr = df[(df["Nutrient"] == carbon) & (df["Rep"] == rep)].copy()
        ymax = dfr[['Variovorax', 'Sphingopyxis', 'Rhodococcus']].values.flatten().max()
        for time in times:
            dft = dfr[(dfr["Time"] == time)].copy()
            # vol_scalor = dft[['Variovorax', 'Sphingopyxis', 'Rhodococcus']].values.flatten().max() / dft["Volume"].max()
            # dft["Volume"] = dft["Volume"] * vol_scalor
            fig = px.line(dft, x='Radius', y=['Variovorax', 'Sphingopyxis', 'Rhodococcus'], title="Population size of " + rep + " with " + carbon + " at " + str(time) + "h", color_discrete_map=color_map, template="ggplot2")
            fig.update_layout(yaxis_range=[0,ymax])
            # fig.show()
            fig.write_image(analysis_dir + "spatial/exploratory/pop_vol_" + rep +"_with_" + carbon + "_at_" + str(time) + "h_" + str(slide_win) + ".pdf", scale=1, width=1000, height=500)

# %%
df = pop_vol_1d_df.copy()
color_map = {'Variovorax / Volume': 'red', 'Sphingopyxis / Volume': 'green', 'Rhodococcus/ Volume': 'blue', 'Relative Volume': 'grey'}
for time in times:
    for rep in ["rep" + str(i+1) for i in range(3)]:
        dfr = df[(df["Time"] == time) & (df["Rep"] == rep)].copy()
        ymax = dfr[['Variovorax', 'Sphingopyxis', 'Rhodococcus']].values.flatten().max()
        for carbon in ["NAG", "Chitopentose", "Chitin"]:
            dft = dfr[(dfr["Nutrient"] == carbon)].copy()
            vol_scalor = dft[['Variovorax / Volume', 'Sphingopyxis / Volume', 'Rhodococcus/ Volume']].values.flatten().max() / dft["Relative Volume"].max()
            dft["Relative Volume"] = dft["Relative Volume"] * vol_scalor
            fig = px.line(dft, x='Radius', y=['Variovorax / Volume', 'Sphingopyxis / Volume', 'Rhodococcus/ Volume', 'Relative Volume'], title="Population size of " + rep + " with " + carbon + " at " + str(time) + "h", color_discrete_map=color_map, template="ggplot2")
            # fig.show()
            fig.write_image(analysis_dir + "spatial/exploratory/pop_rel_vol_" + rep +"_with_" + carbon + "_at_" + str(time) + "h_" + str(slide_win) + ".pdf", scale=1, width=1000, height=500)

# %%
print("The maximum habitat size is: {}".format(pop_den_1d_df["Volume"].max()))

# %% [markdown]
# # Analyze all data

# %%
# import time
# import datetime
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pathlib import Path
# import numpy as np
# import scipy as sp
# import pandas as pd
# from statannotations.Annotator import Annotator
# import matplotlib
# # %matplotlib inline
# matplotlib.rcParams['figure.dpi'] = 200
# matplotlib.rcParams['figure.figsize'] = (5, 5)

# %%
# work_dir = "/Users/feng626/workspace/data/soil1.3/spatial_interactions/"
# result_dir = work_dir + "20220408/"

# # analysis_dir = "./results/{}/".format(datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S.%f"))
# analysis_dir = "./results/{}/".format(datetime.datetime.now().strftime("%Y%m%d"))
# Path(analysis_dir).mkdir(parents=True, exist_ok=True)

# %%
slide_win = 500
pop_den = pd.read_pickle(analysis_dir + "pop_densities.pkl")
pop_vol = pd.read_pickle(analysis_dir + "pop_volumes.pkl")
pop_den_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_den_1d_df_" + str(slide_win) + ".pkl")
delta_pop_den_1d_df = pd.read_pickle(analysis_dir + "spatial/delta_pop_den_1d_df_" + str(slide_win) + ".pkl")
pop_vol_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_vol_1d_df_" + str(slide_win) + ".pkl")
delta_pop_vol_1d_df = pd.read_pickle(analysis_dir + "spatial/delta_pop_vol_1d_df_" + str(slide_win) + ".pkl")

# %%
sns.set_style("white")
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (4, 4)

# %%
pop_vol

# %%
sns.lineplot(data=pop_vol, x="Time", y="Total", hue="Nutrient")
plt.savefig(analysis_dir + "lineplot_total_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()
sns.lineplot(data=pop_vol, x="Time", y="Variovorax", hue="Nutrient")
plt.savefig(analysis_dir + "lineplot_Vp_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()
sns.lineplot(data=pop_vol, x="Time", y="Sphingopyxis", hue="Nutrient")
plt.savefig(analysis_dir + "lineplot_Sf_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()
sns.lineplot(data=pop_vol, x="Time", y="Rhodococcus", hue="Nutrient")
plt.savefig(analysis_dir + "lineplot_Rh_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# %%
sns.lineplot(data=pop_vol, x="Time", y="Total", hue="Nutrient", err_style="bars", ci=66.7)
plt.ylim(0, 8e7)
plt.savefig(analysis_dir + "lineplot_bar_total_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()
sns.lineplot(data=pop_vol, x="Time", y="Variovorax", hue="Nutrient", err_style="bars", ci=66.7)
plt.ylim(0, 5e7)
plt.savefig(analysis_dir + "lineplot_bar_Vp_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()
sns.lineplot(data=pop_vol, x="Time", y="Sphingopyxis", hue="Nutrient", err_style="bars", ci=66.7)
plt.ylim(0, 5e7)
plt.savefig(analysis_dir + "lineplot_bar_Sf_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()
sns.lineplot(data=pop_vol, x="Time", y="Rhodococcus", hue="Nutrient", err_style="bars", ci=66.7)
plt.ylim(0, 5e7)
plt.savefig(analysis_dir + "lineplot_bar_Rh_volume.pdf", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# %%
sns.boxplot(data=pop_vol, x="Time", y="Total", hue="Nutrient")
# plt.show()
plt.savefig(analysis_dir + "boxplot_total_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()
sns.boxplot(data=pop_vol, x="Time", y="Variovorax", hue="Nutrient")
plt.savefig(analysis_dir + "boxplot_Vp_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()
sns.boxplot(data=pop_vol, x="Time", y="Sphingopyxis", hue="Nutrient")
plt.savefig(analysis_dir + "boxplot_Sf_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()
sns.boxplot(data=pop_vol, x="Time", y="Rhodococcus", hue="Nutrient")
plt.savefig(analysis_dir + "boxplot_Rh_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

# %%
sns.boxplot(data=pop_vol, x="Time", y="Total", hue="Nutrient")
plt.yscale("log")
plt.savefig(analysis_dir + "boxplot_total_volume_log.pdf", bbox_inches='tight', dpi=300)
plt.close()
sns.boxplot(data=pop_vol, x="Time", y="Variovorax", hue="Nutrient")
plt.yscale("log")
plt.savefig(analysis_dir + "boxplot_Vp_volume_log.pdf", bbox_inches='tight', dpi=300)
plt.close()
sns.boxplot(data=pop_vol, x="Time", y="Sphingopyxis", hue="Nutrient")
plt.yscale("log")
plt.savefig(analysis_dir + "boxplot_Sf_volume_log.pdf", bbox_inches='tight', dpi=300)
plt.close()
sns.boxplot(data=pop_vol, x="Time", y="Rhodococcus", hue="Nutrient")
plt.yscale("log")
plt.savefig(analysis_dir + "boxplot_Rh_volume_log.pdf", bbox_inches='tight', dpi=300)
plt.close()

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Nutrient"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

pairs=[("Variovorax", "Sphingopyxis"), ("Sphingopyxis", "Rhodococcus"), ("Rhodococcus", "Variovorax")]

# df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "NAG")]
# ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
# plt.ylim(0, 1.1e8)
# plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
# plt.savefig(analysis_dir + "boxplot_NAG_volume.pdf", bbox_inches='tight', dpi=300)
# plt.close()

# df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitopentose")]
# ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
# plt.ylim(0, 1.1e8)
# plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
# plt.savefig(analysis_dir + "boxplot_Chitopentose_volume.pdf", bbox_inches='tight', dpi=300)
# plt.close()

# df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitin")]
# ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
# plt.ylim(0, 1.1e8)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
# plt.savefig(analysis_dir + "boxplot_Chitin_volume.pdf", bbox_inches='tight', dpi=300)
# plt.close()

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

pairs=[
    [(0, "Sphingopyxis"), (0, "Variovorax")], 
    [(0, "Rhodococcus"),(0, "Sphingopyxis")], 
    [(0, "Variovorax"), (0, "Rhodococcus")],
    
    [(11, "Sphingopyxis"), (11, "Variovorax")], 
    [(11, "Rhodococcus"),(11, "Sphingopyxis")], 
    [(11, "Variovorax"), (11, "Rhodococcus")],
    
    [(22, "Sphingopyxis"), (22, "Variovorax")], 
    [(22, "Rhodococcus"),(22, "Sphingopyxis")], 
    [(22, "Variovorax"), (22, "Rhodococcus")],
    
    [(37, "Sphingopyxis"), (37, "Variovorax")], 
    [(37, "Rhodococcus"),(37, "Sphingopyxis")], 
    [(37, "Variovorax"), (37, "Rhodococcus")],
    
    [(66, "Sphingopyxis"), (66, "Variovorax")], 
    [(66, "Rhodococcus"),(66, "Sphingopyxis")], 
    [(66, "Variovorax"), (66, "Rhodococcus")],
    
    [(90, "Sphingopyxis"), (90, "Variovorax")], 
    [(90, "Rhodococcus"),(90, "Sphingopyxis")], 
    [(90, "Variovorax"), (90, "Rhodococcus")],
    ]


df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "NAG")]

hue_plot_params = {
    'data': df,
    'x': 'Time',
    'y': 'Biomass',
    # "order": subcat_order,
    "hue": "Strain",
    # "hue_order": states_order,
    "palette": color_map
}

ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
annotator = Annotator(ax, pairs, **hue_plot_params)
annotator.configure(test="Mann-Whitney").apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_NAG_volume_with_tot_inter_species.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitopentose")]

hue_plot_params = {
    'data': df,
    'x': 'Time',
    'y': 'Biomass',
    # "order": subcat_order,
    "hue": "Strain",
    # "hue_order": states_order,
    "palette": color_map
}

ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
annotator = Annotator(ax, pairs, **hue_plot_params)
annotator.configure(test="Mann-Whitney").apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitopentose_volume_with_tot_inter_species.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitin")]

hue_plot_params = {
    'data': df,
    'x': 'Time',
    'y': 'Biomass',
    # "order": subcat_order,
    "hue": "Strain",
    # "hue_order": states_order,
    "palette": color_map
}

ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
annotator = Annotator(ax, pairs, **hue_plot_params)
annotator.configure(test="Mann-Whitney").apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitin_volume_with_tot_inter_species.pdf", bbox_inches='tight', dpi=300)
plt.close()

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Nutrient"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

pairs=[("Variovorax", "Sphingopyxis"), ("Sphingopyxis", "Rhodococcus"), ("Rhodococcus", "Variovorax")]

# df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "NAG")]
# ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
# plt.ylim(0, 1.1e8)
# plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
# plt.savefig(analysis_dir + "boxplot_NAG_volume.pdf", bbox_inches='tight', dpi=300)
# plt.close()

# df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitopentose")]
# ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
# plt.ylim(0, 1.1e8)
# plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
# plt.savefig(analysis_dir + "boxplot_Chitopentose_volume.pdf", bbox_inches='tight', dpi=300)
# plt.close()

# df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitin")]
# ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
# plt.ylim(0, 1.1e8)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
# plt.savefig(analysis_dir + "boxplot_Chitin_volume.pdf", bbox_inches='tight', dpi=300)
# plt.close()

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

pairs=[
    [(0, "Sphingopyxis"), (37, "Sphingopyxis")], 
    [(0, "Rhodococcus"),(37, "Rhodococcus")], 
    [(0, "Variovorax"), (37, "Variovorax")],
    
    [(22, "Sphingopyxis"), (90, "Sphingopyxis")], 
    [(22, "Rhodococcus"),(90, "Rhodococcus")], 
    [(22, "Variovorax"), (90, "Variovorax")],
    
    [(0, "Sphingopyxis"), (90, "Sphingopyxis")], 
    [(0, "Rhodococcus"),(90, "Rhodococcus")], 
    [(0, "Variovorax"), (90, "Variovorax")],
    ]


df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "NAG")]

hue_plot_params = {
    'data': df,
    'x': 'Time',
    'y': 'Biomass',
    # "order": subcat_order,
    "hue": "Strain",
    # "hue_order": states_order,
    "palette": color_map
}

ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
annotator = Annotator(ax, pairs, **hue_plot_params)
annotator.configure(test="Mann-Whitney").apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_NAG_volume_with_tot_inter_time.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitopentose")]

hue_plot_params = {
    'data': df,
    'x': 'Time',
    'y': 'Biomass',
    # "order": subcat_order,
    "hue": "Strain",
    # "hue_order": states_order,
    "palette": color_map
}

ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
annotator = Annotator(ax, pairs, **hue_plot_params)
annotator.configure(test="Mann-Whitney").apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitopentose_volume_with_tot_inter_time.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitin")]

hue_plot_params = {
    'data': df,
    'x': 'Time',
    'y': 'Biomass',
    # "order": subcat_order,
    "hue": "Strain",
    # "hue_order": states_order,
    "palette": color_map
}

ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
annotator = Annotator(ax, pairs, **hue_plot_params)
annotator.configure(test="Mann-Whitney").apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitin_volume_with_tot_inter_time.pdf", bbox_inches='tight', dpi=300)
plt.close()

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Nutrient"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

pairs=[("Variovorax", "Sphingopyxis"), ("Sphingopyxis", "Rhodococcus"), ("Rhodococcus", "Variovorax")]

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "NAG")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_NAG_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitopentose")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_Chitopentose_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitin")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_Chitin_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "NAG")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_NAG_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitopentose")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_Chitopentose_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitin")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_Chitin_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Nutrient"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

pairs=[("Variovorax", "Sphingopyxis"), ("Sphingopyxis", "Rhodococcus"), ("Rhodococcus", "Variovorax")]

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "NAG")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map, err_style="bars", ci=66.7)
plt.ylim(0, 8e7)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_bar_NAG_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitopentose")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map, err_style="bars", ci=66.7)
plt.ylim(0, 8e7)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_bar_Chitopentose_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitin")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map, err_style="bars", ci=66.7)
plt.ylim(0, 8e7)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_bar_Chitin_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "NAG")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map, err_style="bars", ci=66.7)
plt.ylim(0, 8e7)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_bar_NAG_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitopentose")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map, err_style="bars", ci=66.7)
plt.ylim(0, 8e7)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_bar_Chitopentose_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitin")]
ax = sns.lineplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map, err_style="bars", ci=66.7)
plt.ylim(0, 8e7)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "lineplot_bar_Chitin_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Nutrient"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")

# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(pop_vol_melted, col="Nutrient", margin_titles=True, despine=False, sharey=True, sharex=True)
g.map_dataframe(sns.lineplot, x="Time", y="Biomass", hue="Strain", palette=color_map, err_style="bars", ci=66.7)
g.add_legend(title="Strains")
plt.savefig(analysis_dir + "lineplot_bar_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
pop_vol_melted

# %%
print(sns.color_palette("Paired").as_hex())
sns.color_palette("Paired")

# %%
print(sns.color_palette("colorblind").as_hex())
sns.color_palette("colorblind")

# %%
print(sns.color_palette().as_hex())
sns.color_palette()

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Nutrient"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")

# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}
color_map = {'NAG': '#029e73', 'Chitopentose': '#d55e00', 'Chitin': '#cc78bc'}

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(pop_vol_melted, col="Strain", margin_titles=True, despine=False, sharey=True, sharex=True)
g.map_dataframe(sns.lineplot, x="Time", y="Biomass", hue="Nutrient", palette=color_map, err_style="bars", ci=66.7)
g.add_legend(title="Nutrients")
plt.savefig(analysis_dir + "lineplot_bar_volume_with_tot_alt.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Nutrient"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

pairs=[("Variovorax", "Sphingopyxis"), ("Sphingopyxis", "Rhodococcus"), ("Rhodococcus", "Variovorax")]

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "NAG")]
ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_NAG_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitopentose")]
ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitopentose_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Strain"] != "Total") & (pop_vol_melted["Nutrient"] == "Chitin")]
ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitin_volume.pdf", bbox_inches='tight', dpi=300)
plt.close()

color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "NAG")]
ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Strain", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_NAG_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitopentose")]
ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
plt.legend([],[], frameon=False)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitopentose_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

df = pop_vol_melted[(pop_vol_melted["Nutrient"] == "Chitin")]
ax = sns.boxplot(data=df, x="Time", y="Biomass", hue="Strain", palette=color_map)
plt.ylim(0, 1.1e8)
# annotator = Annotator(ax, pairs, data=df, x="Time", y="Biomass") #, order=order)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()
plt.savefig(analysis_dir + "boxplot_Chitin_volume_with_tot.pdf", bbox_inches='tight', dpi=300)
plt.close()

# %%
df = pop_vol_melted
# for time in times:
for time in [90]:
    for carbon in ["NAG", "Chitopentose", "Chitin"]:
        for species in ["Variovorax", "Sphingopyxis", "Rhodococcus"]:
            print("The average biomass of {} in {} condition at {} hours is {}".format(species, carbon, time, df[(df["Nutrient"] == carbon) & (df["Time"] == time) & (df["Strain"] == species)]["Biomass"].mean()))

# %%
df = pop_vol_1d_df.copy()
df = df[['Time', 'Nutrient', 'Variovorax / Volume', 'Sphingopyxis / Volume', 'Rhodococcus/ Volume', 'Relative Volume', 'Radius']].copy()
df = df[(df["Time"] >22) & (df["Radius"] % (25 * pixel_size) == 0)]
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
# sns.scatterplot(data=df, y="Variovorax / Volume", x="Relative Volume", hue="Nutrient")
sns.scatterplot(data=df, y="Sphingopyxis / Volume", x="Relative Volume", hue="Nutrient")
# sns.scatterplot(data=df, y="Rhodococcus/ Volume", x="Relative Volume", hue="Nutrient")
plt.show()

# %%
df = pop_vol_1d_df.copy()
# df = df[df["Time"] == 37 & (df["Rep"] == 'rep2')]
df["dv_dvol"] = np.gradient(df["Variovorax / Volume"], df["Relative Volume"])
df["ds_dvol"] = np.gradient(df["Sphingopyxis / Volume"], df["Relative Volume"])
df["dr_dvol"] = np.gradient(df["Rhodococcus/ Volume"], df["Relative Volume"])
df["dv_drad"] = np.gradient(df["Variovorax / Volume"], df["Radius"])
df["ds_drad"] = np.gradient(df["Sphingopyxis / Volume"], df["Radius"])
df["dr_drad"] = np.gradient(df["Rhodococcus/ Volume"], df["Radius"])
df["dvol_drad"] = np.gradient(df["Relative Volume"], df["Radius"])
df["Habitat size"] = np.nan
df.loc[df["Relative Volume"] < 0.4, "Habitat size"] = "Small (0.2~0.4)"
df.loc[(df["Relative Volume"] >= 0.5) & (df["Relative Volume"] < 0.7), "Habitat size"] = "Medium (0.5~0.7)"
df.loc[df["Relative Volume"] >= 0.8, "Habitat size"] = "Large (0.8~1.0)"
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
# sns.scatterplot(data=df, y="Variovorax / Volume", x="Relative Volume", hue="Nutrient")
# sns.scatterplot(data=df, y="Sphingopyxis / Volume", x="Relative Volume", hue="Nutrient")
# sns.scatterplot(data=df, y="Rhodococcus/ Volume", x="Relative Volume", hue="Nutrient")

# sns.scatterplot(data=df, y="Variovorax", x="Relative Volume", hue="Nutrient")
sns.scatterplot(data=df, y="Sphingopyxis", x="Volume", hue="Nutrient")
# sns.scatterplot(data=df, y="Rhodococcus", x="Relative Volume", hue="Nutrient")

# sns.scatterplot(data=df, y="dv_dvol", x="ds_dvol", hue="Nutrient")

plt.close()

# %%
df = pop_vol_1d_df.copy()
df["Habitat size"] = np.nan
df.loc[df["Relative Volume"] < 0.3, "Habitat size"] = "Small (0.2~0.4)"
df.loc[(df["Relative Volume"] >= 0.4) & (df["Relative Volume"] < 0.7), "Habitat size"] = "Medium (0.5~0.7)"
df.loc[df["Relative Volume"] >= 0.8, "Habitat size"] = "Large (0.8~1.0)"
# df.loc[df["Relative Volume"] < 0.2, "Habitat size"] = "[0,0.2)"
# df.loc[df["Relative Volume"] < 0.1, "Habitat size"] = "[0,0.1)"
# df.loc[(df["Relative Volume"] >= 0.1) & (df["Relative Volume"] < 0.2), "Habitat size"] = "[0.1,0.2)"
# df.loc[(df["Relative Volume"] >= 0.2) & (df["Relative Volume"] < 0.3), "Habitat size"] = "[0.2,0.3)"
# df.loc[(df["Relative Volume"] >= 0.3) & (df["Relative Volume"] < 0.4), "Habitat size"] = "[0.3,0.4)"
# df.loc[(df["Relative Volume"] >= 0.4) & (df["Relative Volume"] < 0.5), "Habitat size"] = "[0.4,0.5)"
# df.loc[(df["Relative Volume"] >= 0.5) & (df["Relative Volume"] < 0.6), "Habitat size"] = "[0.5,0.6)"
# df.loc[(df["Relative Volume"] >= 0.6) & (df["Relative Volume"] < 0.7), "Habitat size"] = "[0.6,0.7)"
# df.loc[(df["Relative Volume"] >= 0.7) & (df["Relative Volume"] < 0.8), "Habitat size"] = "[0.7,0.8)"
# df.loc[(df["Relative Volume"] >= 0.8) & (df["Relative Volume"] < 0.9), "Habitat size"] = "[0.8,0.9)"
# df.loc[df["Relative Volume"] >= 0.9, "Habitat size"] = "[0.9,1.0)"
# df.loc[df["Relative Volume"] >= 0.8, "Habitat size"] = "[0.8,1.0]"
# df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat size", "Radius", "Relative Volume"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}
df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
# df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat size", "Radius", "Relative Volume"], value_vars=["Variovorax / Volume", "Sphingopyxis / Volume", "Rhodococcus/ Volume"], var_name="Strain", value_name="Biomass / Habitat size")
# color_map = {'Variovorax / Volume': 'red', 'Sphingopyxis / Volume': 'green', 'Rhodococcus/ Volume': 'blue'}

df4plot = df4plot.loc[df4plot["Time"] == 90]
df4plot = df4plot.loc[df4plot["Biomass"] > 0]
# sns.boxplot(data=df4plot[df4plot["Time"] == 90 & (df4plot["Nutrient"] == "NAG")], x="Habitat size", y="Biomass", hue="Strain", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Nutrient"] == "NAG")], x="Habitat size", y="Biomass", hue="Strain", palette=color_map, showfliers = False)
sns.boxplot(data=df4plot[(df4plot["Nutrient"] == "Chitopentose")], x="Habitat size", y="Biomass", hue="Strain", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Nutrient"] == "Chitin")], x="Habitat size", y="Biomass", hue="Strain", palette=color_map, showfliers = False)
# plt.ylim(1, 1.5e4)
# plt.yscale("log")

plt.close()

# %%
df = pop_den_1d_df.copy()
# df = df.loc[((df[["Variovorax", "Sphingopyxis", "Rhodococcus"]] != 0).all(axis=1)) & (df["Rep"] == "rep2")].copy()
df = df.loc[(df[["Variovorax", "Sphingopyxis", "Rhodococcus"]] != 0).all(axis=1)].copy()
df["Habitat size"] = np.nan
# df.loc[df["Relative Volume"] < 0.3, "Habitat size"] = "Small"
df.loc[(df["Relative Volume"] < 0.4) & (df["Relative Volume"] > 0.2), "Habitat size"] = "(0.2~0.4)"
df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.6), "Habitat size"] = "(0.4~0.6)"
df.loc[(df["Relative Volume"] > 0.6) & (df["Relative Volume"] < 0.8), "Habitat size"] = "(0.6~0.8)"
# df.loc[(df["Relative Volume"] > 0.5) & (df["Relative Volume"] < 0.7), "Habitat size"] = "(0.5~0.7)"
# df.loc[df["Relative Volume"] >= 0.8, "Habitat size"] = "Large"
df.loc[df["Relative Volume"] > 0.8, "Habitat size"] = "(0.8~1.0)"
df.loc[df["Relative Volume"] < 0.2, "Habitat size"] = "(0.0~0.2)"
# df.loc[df["Relative Volume"] < 0.1, "Habitat size"] = "[0,0.1)"
# df.loc[(df["Relative Volume"] > 0.1) & (df["Relative Volume"] < 0.2), "Habitat size"] = "[0.1,0.2)"
# df.loc[(df["Relative Volume"] > 0.2) & (df["Relative Volume"] < 0.3), "Habitat size"] = "[0.2,0.3)"
# df.loc[(df["Relative Volume"] > 0.3) & (df["Relative Volume"] < 0.4), "Habitat size"] = "[0.3,0.4)"
# df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.5), "Habitat size"] = "[0.4,0.5)"
# df.loc[(df["Relative Volume"] > 0.5) & (df["Relative Volume"] < 0.6), "Habitat size"] = "[0.5,0.6)"
# df.loc[(df["Relative Volume"] > 0.6) & (df["Relative Volume"] < 0.7), "Habitat size"] = "[0.6,0.7)"
# df.loc[(df["Relative Volume"] > 0.7) & (df["Relative Volume"] < 0.8), "Habitat size"] = "[0.7,0.8)"
# df.loc[(df["Relative Volume"] > 0.8) & (df["Relative Volume"] < 0.9), "Habitat size"] = "[0.8,0.9)"
# df.loc[df["Relative Volume"] > 0.9, "Habitat size"] = "[0.9,1.0)"
# df.loc[df["Relative Volume"] > 0.8, "Habitat size"] = "[0.8,1.0]"
# df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat size", "Radius", "Relative Volume"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Density")
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}
df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Density")
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
# df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat size", "Radius", "Relative Volume"], value_vars=["Variovorax / Volume", "Sphingopyxis / Volume", "Rhodococcus/ Volume"], var_name="Strain", value_name="Density / Habitat size")
# color_map = {'Variovorax / Volume': 'red', 'Sphingopyxis / Volume': 'green', 'Rhodococcus/ Volume': 'blue'}

df4plot = df4plot.loc[(df4plot["Density"] > 0)].copy()
# df4plot = df4plot.loc[df4plot["Time"] == 37]
# sns.boxplot(data=df4plot[df4plot["Time"] == 90 & (df4plot["Nutrient"] == "NAG")], x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Nutrient"] == "NAG")], x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Nutrient"] == "Chitopentose")], x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Nutrient"] == "Chitin")], x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False)
# plt.ylim(1, 1.5e4)
# plt.yscale("log")

matplotlib.rcParams['figure.figsize'] = (4, 4)

pairs=[
    [("(0.2~0.4)", "Sphingopyxis"), ("(0.8~1.0)", "Sphingopyxis")], 
    [("(0.2~0.4)", "Rhodococcus"),("(0.8~1.0)", "Rhodococcus")], 
    [("(0.2~0.4)", "Variovorax"), ("(0.8~1.0)", "Variovorax")],
    
    # [("(0.4~0.6)", "Sphingopyxis"), ("(0.4~0.6)", "Variovorax")], 
    # [("(0.4~0.6)", "Rhodococcus"),("(0.4~0.6)", "Sphingopyxis")], 
    # [("(0.4~0.6)", "Variovorax"), ("(0.4~0.6)", "Rhodococcus")],
    
    # [("(0.6~0.8)", "Sphingopyxis"), ("(0.6~0.8)", "Variovorax")], 
    # [("(0.6~0.8)", "Rhodococcus"),("(0.6~0.8)", "Sphingopyxis")], 
    # [("(0.6~0.8)", "Variovorax"), ("(0.6~0.8)", "Rhodococcus")],
    
    # [("(0.8~1.0)", "Sphingopyxis"), ("(0.8~1.0)", "Variovorax")], 
    # [("(0.8~1.0)", "Rhodococcus"),("(0.8~1.0)", "Sphingopyxis")], 
    # [("(0.8~1.0)", "Variovorax"), ("(0.8~1.0)", "Rhodococcus")],
    ]
# formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

for time in [11, 37, 90]:
    for carbon in ["NAG", "Chitopentose", "Chitin"]:
        df4plot2 = df4plot.loc[(df4plot["Time"] == time) & (df4plot["Nutrient"] == carbon)].copy()
        hue_plot_params = {
            'data': df4plot2,
            'x': 'Habitat size',
            'y': 'Density',
            # "order": subcat_order,
            "order" : ["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"],
            "hue": "Strain",
            # "hue_order": states_order,
            "palette": color_map
        }
        ax = sns.boxplot(data=df4plot2, x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False, order=["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"])
        # sns.boxplot(data=df4plot2, x="Habitat size", y="Density", hue="Strain", palette=color_map)
        annotator = Annotator(ax, pairs, **hue_plot_params)
        # annotator.configure(test="Mann-Whitney").set_custom_annotations(formatted_pvalues).apply_and_annotate()
        annotator.configure(test="Mann-Whitney").apply_and_annotate()
        plt.legend([],[], frameon=False)
        plt.savefig(analysis_dir + "density_vs_habitate_size_" + carbon + "_at_" + str(time) + "_selective.pdf", bbox_inches='tight', dpi=300)
        plt.close()



df4plot2 = df4plot.loc[(df4plot["Density"] > 0) & (df4plot["Time"].isin([11,37,90]))].copy()
# df4plot2 = df4plot.loc[(df4plot["Density"] > 0) & (df4plot["Time"].isin([11,37,90])) & (df["Rep"] == "rep2")].copy()
# remap = {0: "Early", 11: "Early", 22:"Middle", 37:"Middle", 66:"Late", 90:"Late"}
# df4plot2 = df4plot.replace({"Time": remap})

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(df4plot2, col="Nutrient",  row="Time", margin_titles=True, despine=False, sharey=True, sharex=True)
g.map_dataframe(sns.boxplot, x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False, order=["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"])
# g.map_dataframe(sns.boxplot, x="Habitat size", y="Density", hue="Strain", palette=color_map, order=["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"])
g.add_legend()
plt.savefig(analysis_dir + "density_vs_habitate_size_all.pdf", bbox_inches='tight', dpi=300)

# %%
df = pop_den_1d_df.copy()
df = df.loc[(df[["Variovorax", "Sphingopyxis", "Rhodococcus"]] != 0).all(axis=1)].copy()
df["Habitat size"] = np.nan
df.loc[(df["Relative Volume"] < 0.4) & (df["Relative Volume"] > 0.2), "Habitat size"] = "(0.2~0.4)"
df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.6), "Habitat size"] = "(0.4~0.6)"
df.loc[(df["Relative Volume"] > 0.6) & (df["Relative Volume"] < 0.8), "Habitat size"] = "(0.6~0.8)"
df.loc[df["Relative Volume"] > 0.8, "Habitat size"] = "(0.8~1.0)"
df.loc[df["Relative Volume"] < 0.2, "Habitat size"] = "(0.0~0.2)"
df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Density")
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

df4plot = df4plot.loc[(df4plot["Density"] > 0)].copy()

matplotlib.rcParams['figure.figsize'] = (4, 4)

pairs=[
    [("(0.2~0.4)", "Sphingopyxis"), ("(0.2~0.4)", "Variovorax")], 
    [("(0.2~0.4)", "Rhodococcus"),("(0.2~0.4)", "Sphingopyxis")], 
    [("(0.2~0.4)", "Variovorax"), ("(0.2~0.4)", "Rhodococcus")],
    
    [("(0.4~0.6)", "Sphingopyxis"), ("(0.4~0.6)", "Variovorax")], 
    [("(0.4~0.6)", "Rhodococcus"),("(0.4~0.6)", "Sphingopyxis")], 
    [("(0.4~0.6)", "Variovorax"), ("(0.4~0.6)", "Rhodococcus")],
    
    [("(0.6~0.8)", "Sphingopyxis"), ("(0.6~0.8)", "Variovorax")], 
    [("(0.6~0.8)", "Rhodococcus"),("(0.6~0.8)", "Sphingopyxis")], 
    [("(0.6~0.8)", "Variovorax"), ("(0.6~0.8)", "Rhodococcus")],
    
    [("(0.8~1.0)", "Sphingopyxis"), ("(0.8~1.0)", "Variovorax")], 
    [("(0.8~1.0)", "Rhodococcus"),("(0.8~1.0)", "Sphingopyxis")], 
    [("(0.8~1.0)", "Variovorax"), ("(0.8~1.0)", "Rhodococcus")],
    ]

for time in times: # [11, 37, 90]:
    for carbon in ["NAG", "Chitopentose", "Chitin"]:
        df4plot2 = df4plot.loc[(df4plot["Time"] == time) & (df4plot["Nutrient"] == carbon)].copy()
        hue_plot_params = {
            'data': df4plot2,
            'x': 'Habitat size',
            'y': 'Density',
            # "order": subcat_order,
            "order" : ["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"],
            "hue": "Strain",
            # "hue_order": states_order,
            "palette": color_map
        }
        ax = sns.boxplot(data=df4plot2, x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False, order=["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"])
        # sns.boxplot(data=df4plot2, x="Habitat size", y="Density", hue="Strain", palette=color_map)
        annotator = Annotator(ax, pairs, **hue_plot_params)
        annotator.configure(test="Mann-Whitney").apply_and_annotate()
        plt.legend([],[], frameon=False)
        plt.savefig(analysis_dir + "density_vs_habitate_size_" + carbon + "_at_" + str(time) + ".pdf", bbox_inches='tight', dpi=300)
        plt.close()



# df4plot2 = df4plot.loc[(df4plot["Density"] > 0) & (df4plot["Time"].isin([11,37,90]))].copy()
df4plot2 = df4plot.loc[(df4plot["Density"] > 0) & (df4plot["Time"].isin(times))].copy()

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(df4plot2, col="Nutrient",  row="Time", margin_titles=True, despine=False, sharey=True, sharex=True)
g.map_dataframe(sns.boxplot, x="Habitat size", y="Density", hue="Strain", palette=color_map, showfliers = False, order=["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"])
# g.map_dataframe(sns.boxplot, x="Habitat size", y="Density", hue="Strain", palette=color_map, order=["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"])
g.add_legend()
plt.savefig(analysis_dir + "density_vs_habitate_size_all_times.pdf", bbox_inches='tight', dpi=300)

# %%
df = pop_vol_1d_df.copy()
df["Habitat position"] = np.nan
df.loc[df["Radius"] < (1500 * pixel_size), "Habitat position"] = "Left"
df.loc[(df["Radius"] >= (1500 * pixel_size)) & (df["Radius"] < (3000 * pixel_size)), "Habitat position"] = "Middle"
df.loc[df["Radius"] >= (3000 * pixel_size), "Habitat position"] = "Right"
df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat position", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

df4plot = df4plot.loc[df4plot["Biomass"] > 0]


for time in [11, 37, 90]:
    for carbon in ["NAG", "Chitopentose", "Chitin"]:
        df4plot2 = df4plot.loc[df4plot["Time"] == time & (df4plot["Nutrient"] == carbon)]
        sns.boxplot(data=df4plot2, x="Habitat position", y="Biomass", hue="Strain", palette=color_map, showfliers = False)
        plt.legend([],[], frameon=False)
        plt.yscale("log")
        plt.savefig(analysis_dir + "biomass_vs_habitate_position_" + carbon + "_at_" + str(time) + "pdf", bbox_inches='tight', dpi=300)
        plt.close()

# %%
df = pop_vol_1d_df.copy()
df["Habitat position"] = np.nan
df.loc[df["Radius"] < (1500 * pixel_size), "Habitat position"] = "Left"
df.loc[(df["Radius"] >= (1500 * pixel_size)) & (df["Radius"] < (3000 * pixel_size)), "Habitat position"] = "Middle"
df.loc[df["Radius"] >= (3000 * pixel_size), "Habitat position"] = "Right"
df4plot = pd.melt(df, id_vars=["Time", "Nutrient", "Habitat position", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Strain", value_name="Biomass")
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}

df4plot = df4plot.loc[df4plot["Biomass"] > 0]


for time in [0, 22, 66]:
    for carbon in ["NAG", "Chitopentose", "Chitin"]:
        df4plot2 = df4plot.loc[df4plot["Time"] == time & (df4plot["Nutrient"] == carbon)]
        sns.boxplot(data=df4plot2, x="Habitat position", y="Biomass", hue="Strain", palette=color_map, showfliers = False)
        plt.legend([],[], frameon=False)
        plt.yscale("log")
        plt.savefig(analysis_dir + "biomass_vs_habitate_position_" + carbon + "_at_" + str(time) + "pdf", bbox_inches='tight', dpi=300)
        plt.close()

# %%
df = pop_den_1d_df.copy()
# df = df[df["Time"] == 90 & (df["Rep"] == 'rep2') & (df["Nutrient"] == 'NAG')]
df = df[df["Time"] == 90 & (df["Rep"] == 'rep2')]
# df["Time"] = df["Time"].astype(str) + " h"
# df["Time"] = (df["Time"]) / 5 + 2
fig = px.scatter_ternary(df, 
                        a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                        color="Relative Volume", symbol="Nutrient")
fig.update_coloraxes(colorbar_x=1.0, colorbar_y=0.52)
# fig.update_layout(legend= {'itemsizing': 'trace'})
fig.update_layout(legend_x=0.8, legend_y=1.0)
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
# fig.write_image(analysis_dir + "test.pdf")
fig.show()

# %%
pop_vol[(pop_vol["Time"] == 90) & (pop_vol["Nutrient"] == "NAG")][["Variovorax", "Sphingopyxis", "Rhodococcus"]].mean(axis=0)
test = pop_vol[(pop_vol["Time"] == 90) & (pop_vol["Nutrient"] == "NAG")][["Variovorax", "Sphingopyxis", "Rhodococcus"]]

# %%
sp.stats.f_oneway(test["Variovorax"][:2], test["Sphingopyxis"][:2], test["Rhodococcus"][:2])

# %%
np.nanmean([np.array([np.nan]), np.array([1]), np.array([1]), np.array([1])])
