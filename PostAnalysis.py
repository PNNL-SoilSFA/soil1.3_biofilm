# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: biofilm (3.12.11)
#     language: python
#     name: python3
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
import kaleido
import plotly.express as px
import plotly.graph_objs as go
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%
from dfply import *
import pingouin as pg


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
def random_crop_coords(img, crop_size = 960):
    img_sizes = img.shape # different lenght in each dimension
    top = np.random.randint(0, img_sizes[0] - crop_size)
    left = np.random.randint(0, img_sizes[1] - crop_size)
    return(top, left)
def crop(img, top, left, crop_size = 960):
    crop = img[top:top + crop_size, left:left + crop_size]
    return(crop)


# %%

# %%
work_dir = "./data/soil1.3/spatial_interactions/"
result_dir = work_dir + "20220408/"

# analysis_dir = "./results/{}/".format(datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S.%f"))
# analysis_dir = "./results/{}/".format(datetime.datetime.now().strftime("%Y%m%d"))
analysis_dir = "./results/20260321/"
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
                # ['11.19.21_20', 'Chitin', 22], # outlier
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
                # ["02.11.22_3PM", "Nag", 90], # outlier
                # ["02.07.22_10PM", "Chito5", 0], # outlier
                ["02.08.22_9AM", "Chito5", 11],
                ["02.08.22_8PM", "Chito5", 22],
                ["02.09.22_11AM", "Chito5", 37],
                ["02.10.22_4PM", "Chito5", 66],
                # ["02.11.22_4PM", "Chito5", 90], # outlier
                ["02.07.22_11PM", "Chitin", 0],
                # ["02.08.22_10AM", "Chitin", 11], # outlier
                ["02.08.22_9PM", "Chitin", 22],
                ["02.09.22_12PM", "Chitin", 37],
                ["02.10.22_5PM", "Chitin", 66],
                # ["02.11.22_5PM", "Chitin", 90] # outlier
                ]


# %% [markdown]
# ## Temporal dynamics

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
            sample_carbon_name = "N-acetylglucosamine"
            # sample_carbon_name = "NAG"
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

pop_den_df = pd.DataFrame(pop_densities, columns = ["Rep", "Growth Substrate", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Volume"])
pop_vol_df = pd.DataFrame(pop_volumes, columns = ["Rep", "Growth Substrate", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Volume"])
pop_vol_df["Total"] = pop_vol_df["Variovorax"] + pop_vol_df["Sphingopyxis"] + pop_vol_df["Rhodococcus"]
pop_vol_df["Variovorax / Volume"] = pop_vol_df["Variovorax"] / pop_vol_df["Volume"]
pop_vol_df["Sphingopyxis / Volume"] = pop_vol_df["Sphingopyxis"] / pop_vol_df["Volume"]
pop_vol_df["Rhodococcus / Volume"] = pop_vol_df["Rhodococcus"] / pop_vol_df["Volume"]

pop_vol_df["Sample"] = pop_vol_df["Growth Substrate"] + "_" + pop_vol_df["Rep"]
pop_den_df["Sample"] = pop_den_df["Growth Substrate"] + "_" + pop_den_df["Rep"]

pop_den_df.to_pickle(analysis_dir + "pop_densities_wo_outliers.pkl")
pop_vol_df.to_pickle(analysis_dir + "pop_volumes_wo_outliers.pkl")


# %%
pop_den_df = pd.read_pickle(analysis_dir + "pop_densities_wo_outliers.pkl")
pop_vol_df = pd.read_pickle(analysis_dir + "pop_volumes_wo_outliers.pkl")


# %%
df_anova = pd.melt(pop_vol_df, id_vars=["Sample", "Growth Substrate", "Time", "Rep", "Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Biomass")
df_anova.head()


# %%
df_anova.to_csv(analysis_dir + "dataframe_for_anova_table2_wo_outliers.tsv", sep='\t')

# %%
anova3 = df_anova.anova(dv='Biomass', between=['Growth Substrate', 'Species', 'Time'], ss_type=3)#.round(5)
anova3.to_csv(analysis_dir + "anova3_wo_outliers.csv")
anova3


# %%
df = pop_den_df.copy()
# df["Time"] = df["Time"].astype(str) + " h"
fig = px.scatter_ternary(df, 
                         a='Variovorax', b='Sphingopyxis', c='Rhodococcus', 
                         color="Time", symbol="Growth Substrate", color_continuous_scale=px.colors.sequential.turbid)
fig.update_traces(marker={'size': 15})
fig.update_coloraxes(colorbar_x=1.03, colorbar_y=0.52)
fig.update_layout(legend_x=0.8, legend_y=1.0)
fig.update_ternaries(aaxis_color="red", baxis_color="green", caxis_color="blue")
fig.show()
fig.write_image(analysis_dir + "rep123_all_reps_alt_cont_wo_outliers.pdf")


# %% [markdown]
# ## Spatial dynamics

# %%
# Get the sampling axis data
slide_win = 480
trials = 100

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

        # print("Processing sample: ", sample)
        # print("Image shape: ", rgb_array.shape)
        
        mask_vol = slide_sum_y_vol(mask_array, slide_win) * pixel_area
        # print("Mask volume shape: ", mask_vol.shape)
        pop_vol = slide_sum_y_vol(rgb_array, slide_win) * pixel_area
        # print("Population volume shape: ", pop_vol.shape)
        pop_den = get_pop_density(slide_sum_y_vol(rgb_array, slide_win))
        # print("Population density shape: ", pop_den.shape)

        if sample[1] == "Nag":
            sample_carbon_name = "N-acetylglucosamine"
            # sample_carbon_name = "NAG"
        elif sample[1] == "Chito5":
            sample_carbon_name = "Chitopentose"
        elif sample[1] == "Chitin":
            sample_carbon_name = "Chitin"
        else:
            sample_carbon_name = "Unknown"
        for trial in range(trials):
            # Random select non duplicate window positions from available positions
            window_positions = np.random.choice(len(mask_vol), size=int(len(mask_vol)/slide_win), replace=False)
            for i in window_positions:
                pop_volumes_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_vol[i,0], pop_vol[i,1], pop_vol[i,2], i * pixel_size, mask_vol[i], i, trial])
                pop_densities_1d.append([folder.split('_')[2], sample_carbon_name, sample[2], pop_den[i,0], pop_den[i,1], pop_den[i,2], i * pixel_size, mask_vol[i], i, trial])


# %%
spatial_dir = analysis_dir + "spatial/"
Path(spatial_dir).mkdir(parents=True, exist_ok=True)


# %%
pop_vol_1d_df = pd.DataFrame(pop_volumes_1d, columns = ["Rep", "Growth Substrate", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Radius", "Volume", "window_position", "Trial"])
pop_vol_1d_df["Relative Volume"] = pop_vol_1d_df["Volume"] / pop_vol_1d_df["Volume"].max()
pop_vol_1d_df["Variovorax / Volume"] = pop_vol_1d_df["Variovorax"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df["Sphingopyxis / Volume"] = pop_vol_1d_df["Sphingopyxis"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df["Rhodococcus/ Volume"] = pop_vol_1d_df["Rhodococcus"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df["Total"] = pop_vol_1d_df["Variovorax"] + pop_vol_1d_df["Sphingopyxis"] + pop_vol_1d_df["Rhodococcus"]
pop_vol_1d_df["Total / Volume"] = pop_vol_1d_df["Total"] / pop_vol_1d_df["Volume"]
pop_vol_1d_df.to_pickle(analysis_dir + "spatial/pop_vol_1d_df_" + str(slide_win) + "_wo_outliers.pkl")


pop_den_1d_df = pd.DataFrame(pop_densities_1d, columns = ["Rep", "Growth Substrate", "Time", "Variovorax", "Sphingopyxis", "Rhodococcus", "Radius", "Volume", "window_position", "Trial"])
pop_den_1d_df["Relative Volume"] = pop_den_1d_df["Volume"] / pop_den_1d_df["Volume"].max()
# pop_den_1d_df["Variovorax / Volume"] = pop_den_1d_df["Variovorax"] / pop_den_1d_df["Volume"]
# pop_den_1d_df["Sphingopyxis / Volume"] = pop_den_1d_df["Sphingopyxis"] / pop_den_1d_df["Volume"]
# pop_den_1d_df["Rhodococcus/ Volume"] = pop_den_1d_df["Rhodococcus"] / pop_den_1d_df["Volume"]
pop_den_1d_df.to_pickle(analysis_dir + "spatial/pop_den_1d_df_" + str(slide_win) + "_wo_outliers.pkl")


# %%
slide_win = 480
pop_vol_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_vol_1d_df_" + str(slide_win) + "_wo_outliers.pkl")
pop_den_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_den_1d_df_" + str(slide_win) + "_wo_outliers.pkl")


# %%
pop_vol_1d_df

# %% [markdown]
# # Analyze all data

# %%
slide_win = 480
pop_den = pd.read_pickle(analysis_dir + "pop_densities_wo_outliers.pkl")
pop_vol = pd.read_pickle(analysis_dir + "pop_volumes_wo_outliers.pkl")
pop_den_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_den_1d_df_" + str(slide_win) + "_wo_outliers.pkl")
pop_vol_1d_df = pd.read_pickle(analysis_dir + "spatial/pop_vol_1d_df_" + str(slide_win) + "_wo_outliers.pkl")


# %%
sns.set_style("white")
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (4, 4)


# %% [markdown]
# ### Plotting

# %% [markdown]
# #### Temporal dynamics plots

# %%
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Growth Substrate"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Biomass")

# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(pop_vol_melted, col="Growth Substrate", margin_titles=True, despine=False, sharey=True, sharex=True)
g.map_dataframe(sns.lineplot, x="Time", y="Biomass", hue="Species", palette=color_map, err_style="bars", ci=66.7)
g.add_legend(title="Species")
plt.savefig(analysis_dir + "lineplot_bar_volume_with_tot_wo_outliers.pdf", bbox_inches='tight', dpi=300)
plt.show()


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
pop_vol_melted = pd.melt(pop_vol, id_vars=["Time", "Growth Substrate"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Biomass")

# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}
color_map = {'N-acetylglucosamine': '#029e73', 'Chitopentose': '#d55e00', 'Chitin': '#cc78bc'}

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(pop_vol_melted, col="Species", margin_titles=True, despine=False, sharey=True, sharex=True)
g.map_dataframe(sns.lineplot, x="Time", y="Biomass", hue="Growth Substrate", palette=color_map, err_style="bars", ci=66.7)
g.add_legend(title="Growth Substrates")
plt.savefig(analysis_dir + "lineplot_bar_volume_with_tot_alt_wo_outliers.pdf", bbox_inches='tight', dpi=300)
plt.show()


# %%
df = pop_vol_melted
# for time in times:
for time in [90]:
    for carbon in ["N-acetylglucosamine", "Chitopentose", "Chitin"]:
        for species in ["Variovorax", "Sphingopyxis", "Rhodococcus"]:
            print("The average biomass of {} in {} condition at {} hours is {}".format(species, carbon, time, df[(df["Growth Substrate"] == carbon) & (df["Time"] == time) & (df["Species"] == species)]["Biomass"].mean()))


# %% [markdown]
# #### ANOVA analyses

# %%
pop_vol_1d_df

# %%
df = pop_vol_1d_df[(pop_vol_1d_df["Trial"] == 0)].copy()
# df = pop_vol_1d_df[(pop_vol_1d_df["Trial"] == 0) & (pop_vol_1d_df["Time"] == 90)].copy()
df["Habitat Size"] = ""
# df.loc[(df["Relative Volume"] < 0.4) & (df["Relative Volume"] > 0.2), "Habitat Size"] = "(0.2~0.4)"
# df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.6), "Habitat Size"] = "(0.4~0.6)"
# df.loc[(df["Relative Volume"] > 0.6) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "(0.6~0.8)"
df.loc[(df["Relative Volume"] >= 0.4) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "(0.4~0.8)"
df.loc[df["Relative Volume"] >= 0.8, "Habitat Size"] = "(0.8~1.0)"
# df.loc[df["Relative Volume"] < 0.2, "Habitat Size"] = "(0.0~0.2)"
df.loc[df["Relative Volume"] < 0.4, "Habitat Size"] = "(0.0~0.4)"
df4plot = pd.melt(df, id_vars=["Time", "Growth Substrate", "Habitat Size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Biomass")
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}

df4plot


# %%
df4plot2 = df4plot[df4plot["Habitat Size"] != "(0.4~0.8)"].copy()

# %%
anova3 = df4plot.anova(dv='Biomass', between=['Growth Substrate', "Habitat Size", "Species"], ss_type=3)
anova3


# %%
anova3 = df4plot2.anova(dv='Biomass', between=['Growth Substrate', "Habitat Size", "Species"], ss_type=3)
anova3


# %%
anova4 = df4plot.anova(dv='Biomass', between=['Growth Substrate', "Habitat Size", "Species", "Time"], ss_type=3)
anova4


# %%
anova4 = df4plot2.anova(dv='Biomass', between=['Growth Substrate', "Habitat Size", "Species", "Time"], ss_type=3)
anova4


# %%
all_anova3s = []
for trial in pop_vol_1d_df["Trial"].unique():
    df = pop_vol_1d_df[(pop_vol_1d_df["Trial"] == trial)].copy()
    df["Habitat Size"] = ""
    df.loc[(df["Relative Volume"] >= 0.4) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "(0.4~0.8)"
    df.loc[df["Relative Volume"] >= 0.8, "Habitat Size"] = "(0.8~1.0)"
    df.loc[df["Relative Volume"] < 0.4, "Habitat Size"] = "(0.0~0.4)"
    df4anova = pd.melt(df, id_vars=["Time", "Growth Substrate", "Habitat Size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Biomass")
    # df4anova = df4anova[df4anova["Habitat Size"] != "(0.4~0.8)"].copy()
    anova3 = df4anova.anova(dv='Biomass', between=['Growth Substrate', "Habitat Size", "Species"], ss_type=3)
    # print("Trial: ", trial)
    # print(anova3)
    anova3["Trial"] = trial
    all_anova3s.append(anova3)
all_anova3s_df_all_sizes = pd.concat(all_anova3s)
all_anova3s_df_all_sizes.to_csv(analysis_dir + "anova3_spatial_anova_wo_outliers_all_sizes.csv")
all_anova3s_df_all_sizes

# %%
all_anova3s_df_all_sizes.groupby("Source").agg({"SS": ["median", "mean"], "DF": ["median"], "MS": ["median", "mean"], "F": ["median", "mean"], "p-unc": ["max", "mean", "median", "min"], "np2": ["max", "median", "mean", "min"], })

# %%
all_anova3s_df_all_sizes.groupby("Source").agg({"SS": ["median", "mean"], "DF": ["median"], "MS": ["median", "mean"], "F": ["median", "mean"], "p-unc": ["max", "mean", "median", "min"], "np2": ["max", "median", "mean", "min"], }).to_csv(analysis_dir + "anova3_spatial_anova_summary_wo_outliers_all_sizes.csv")

# %%
all_anova3s_df_all_sizes.groupby("Source").agg({"SS": "mean", "DF": "mean", "MS": "median", "F": "mean", "p-unc": "mean", "np2": "mean"}).to_csv(analysis_dir + "anova3_spatial_anova_mean_wo_outliers_all_sizes.csv")

# %%
sampled_data_for_anova_table3 = []
all_anova3s = []
for trial in pop_vol_1d_df["Trial"].unique():
    df = pop_vol_1d_df[(pop_vol_1d_df["Trial"] == trial)].copy()
    df["Habitat Size"] = ""
    df.loc[(df["Relative Volume"] >= 0.4) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "(0.4~0.8)"
    df.loc[df["Relative Volume"] >= 0.8, "Habitat Size"] = "(0.8~1.0)"
    df.loc[df["Relative Volume"] < 0.4, "Habitat Size"] = "(0.0~0.4)"
    df4anova = pd.melt(df, id_vars=["Time", "Growth Substrate", "Habitat Size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Biomass")
    df4anova = df4anova[df4anova["Habitat Size"] != "(0.4~0.8)"].copy()
    sampled_data_for_anova_table3.append(df4anova)
    anova3 = df4anova.anova(dv='Biomass', between=['Growth Substrate', "Habitat Size", "Species"], ss_type=3)
    # print("Trial: ", trial)
    # print(anova3)
    anova3["Trial"] = trial
    all_anova3s.append(anova3)
all_anova3s_df = pd.concat(all_anova3s)
sampled_data_for_anova_table3_df = pd.concat(sampled_data_for_anova_table3)
sampled_data_for_anova_table3_df.to_csv(analysis_dir + "dataframe_randomly_sampled_for_anova_table3_without_outliers.csv")
# all_anova3s_df.to_csv(analysis_dir + "anova3_spatial_anova_wo_outliers.csv")
all_anova3s_df

# %%
all_anova3s_df.groupby("Source").agg({"SS": ["median", "mean"], "DF": ["median"], "MS": ["median", "mean"], "F": ["median", "mean"], "p-unc": ["max", "mean", "median", "min"], "np2": ["max", "median", "mean", "min"], })

# %%
all_anova3s_df.groupby("Source").agg({"SS": ["median", "mean"], "DF": ["median"], "MS": ["median", "mean"], "F": ["median", "mean"], "p-unc": ["max", "mean", "median", "min"], "np2": ["max", "median", "mean", "min"], }).to_csv(analysis_dir + "anova3_spatial_anova_summary_wo_outliers.csv")

# %%
all_anova3s_df.groupby("Source").agg({"SS": "mean", "DF": "mean", "MS": "median", "F": "mean", "p-unc": "mean", "np2": "mean"}).to_csv(analysis_dir + "anova3_spatial_anova_mean_wo_outliers.csv")

# %%

# %% [markdown]
# #### Violin plots

# %%
df = pop_den_1d_df.copy()
# df = df.loc[((df[["Variovorax", "Sphingopyxis", "Rhodococcus"]] != 0).all(axis=1)) & (df["Rep"] == "rep2")].copy()
df = df.loc[(df[["Variovorax", "Sphingopyxis", "Rhodococcus"]] != 0).all(axis=1)].copy()
df["Habitat Size"] = ""
df.loc[df["Relative Volume"] <= 0.4, "Habitat Size"] = "Pore throat"
df.loc[df["Relative Volume"] >= 0.8, "Habitat Size"] = "Pore body"
df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "Intermediate"
# df.loc[df["Relative Volume"] < 0.3, "Habitat Size"] = "Small"
# df.loc[(df["Relative Volume"] < 0.4) & (df["Relative Volume"] > 0.2), "Habitat Size"] = "(0.2~0.4)"
# df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.6), "Habitat Size"] = "(0.4~0.6)"
# df.loc[(df["Relative Volume"] > 0.6) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "(0.6~0.8)"
# df.loc[(df["Relative Volume"] > 0.5) & (df["Relative Volume"] < 0.7), "Habitat Size"] = "(0.5~0.7)"
# df.loc[df["Relative Volume"] >= 0.8, "Habitat Size"] = "Large"
# df.loc[df["Relative Volume"] > 0.8, "Habitat Size"] = "(0.8~1.0)"
# df.loc[df["Relative Volume"] < 0.2, "Habitat Size"] = "(0.0~0.2)"
# df.loc[df["Relative Volume"] < 0.1, "Habitat Size"] = "[0,0.1)"
# df.loc[(df["Relative Volume"] > 0.1) & (df["Relative Volume"] < 0.2), "Habitat Size"] = "[0.1,0.2)"
# df.loc[(df["Relative Volume"] > 0.2) & (df["Relative Volume"] < 0.3), "Habitat Size"] = "[0.2,0.3)"
# df.loc[(df["Relative Volume"] > 0.3) & (df["Relative Volume"] < 0.4), "Habitat Size"] = "[0.3,0.4)"
# df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.5), "Habitat Size"] = "[0.4,0.5)"
# df.loc[(df["Relative Volume"] > 0.5) & (df["Relative Volume"] < 0.6), "Habitat Size"] = "[0.5,0.6)"
# df.loc[(df["Relative Volume"] > 0.6) & (df["Relative Volume"] < 0.7), "Habitat Size"] = "[0.6,0.7)"
# df.loc[(df["Relative Volume"] > 0.7) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "[0.7,0.8)"
# df.loc[(df["Relative Volume"] > 0.8) & (df["Relative Volume"] < 0.9), "Habitat Size"] = "[0.8,0.9)"
# df.loc[df["Relative Volume"] > 0.9, "Habitat Size"] = "[0.9,1.0)"
# df.loc[df["Relative Volume"] > 0.8, "Habitat Size"] = "[0.8,1.0]"
# df4plot = pd.melt(df, id_vars=["Time", "Growth Substrate", "Habitat Size", "Radius", "Relative Volume"], value_vars=["Total", "Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Relative abundance")
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue', 'Total': 'grey'}
df4plot = pd.melt(df, id_vars=["Time", "Growth Substrate", "Habitat Size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Relative abundance")
# color_map = {'Variovorax': 'red', 'Sphingopyxis': 'green', 'Rhodococcus': 'blue'}
# df4plot = pd.melt(df, id_vars=["Time", "Growth Substrate", "Habitat Size", "Radius", "Relative Volume"], value_vars=["Variovorax / Volume", "Sphingopyxis / Volume", "Rhodococcus/ Volume"], var_name="Species", value_name="Relative abundance / Habitat Size")
# color_map = {'Variovorax / Volume': 'red', 'Sphingopyxis / Volume': 'green', 'Rhodococcus/ Volume': 'blue'}

df4plot = df4plot.loc[(df4plot["Relative abundance"] > 0) & (df4plot["Habitat Size"] != "Intermediate")].copy()
# df4plot = df4plot.loc[df4plot["Time"] == 37]
# sns.boxplot(data=df4plot[df4plot["Time"] == 90 & (df4plot["Growth Substrate"] == "NAG")], x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Growth Substrate"] == "NAG")], x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Growth Substrate"] == "Chitopentose")], x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map, showfliers = False)
# sns.boxplot(data=df4plot[(df4plot["Growth Substrate"] == "Chitin")], x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map, showfliers = False)
# plt.ylim(1, 1.5e4)
# plt.yscale("log")

matplotlib.rcParams['figure.figsize'] = (4, 4)

pairs=[
    # [("Pore throat", "Sphingopyxis"), ("Pore body", "Sphingopyxis")], 
    # [("Pore throat", "Rhodococcus"),("Pore body", "Rhodococcus")], 
    # [("Pore throat", "Variovorax"), ("Pore body", "Variovorax")],
    [("Sphingopyxis", "Pore throat"), ("Sphingopyxis", "Pore body")], 
    [("Rhodococcus", "Pore throat"),("Rhodococcus", "Pore body")], 
    [("Variovorax", "Pore throat"), ("Variovorax", "Pore body")],
    ]
# formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

for time in [11, 37, 90]:
    for carbon in ["N-acetylglucosamine", "Chitopentose", "Chitin"]:
        df4plot2 = df4plot.loc[(df4plot["Time"] == time) & (df4plot["Growth Substrate"] == carbon)].copy()
        hue_plot_params = {
            'data': df4plot2,
            # 'x': 'Habitat Size',
            'x': 'Species',
            'y': 'Relative abundance',
            "order": ["Variovorax", "Sphingopyxis", "Rhodococcus"], #subcat_order,
            # "order" : ["Pore throat", "Pore body"],
            "hue": "Habitat Size",
            # "hue_order": states_order,
            # "palette": color_map
        }
        # ax = sns.boxplot(data=df4plot2, x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map, showfliers = False, order=["Pore throat", "Pore body"])
        ax = sns.violinplot(data=df4plot2, 
                        # x="Habitat Size", 
                        x="Species", 
                        y="Relative abundance", 
                        # hue="Species", 
                        hue="Habitat Size", 
                        # palette = color_map, 
                        # showfliers = False, 
                        # order=["Pore throat", "Pore body"],
                        order = ["Variovorax", "Sphingopyxis", "Rhodococcus"],
                        )
        # sns.boxplot(data=df4plot2, x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map)
        # annotator = Annotator(ax, pairs, **hue_plot_params)
        # annotator.configure(test="Mann-Whitney").set_custom_annotations(formatted_pvalues).apply_and_annotate()
        # annotator.configure(test="Mann-Whitney").apply_and_annotate()
        plt.legend([],[], frameon=False)
        # plt.savefig(analysis_dir + "relative_abundance_vs_habitate_size_" + carbon + "_at_" + str(time) + "_selective_wo_outliers.pdf", bbox_inches='tight', dpi=300)
        plt.close()



df4plot2 = df4plot.loc[(df4plot["Relative abundance"] > 0) & (df4plot["Time"].isin([11,37,90]))].copy()
# df4plot2 = df4plot.loc[(df4plot["Relative abundance"] > 0) & (df4plot["Time"].isin([11,37,90])) & (df["Rep"] == "rep2")].copy()
# remap = {0: "Early", 11: "Early", 22:"Middle", 37:"Middle", 66:"Late", 90:"Late"}
# df4plot2 = df4plot.replace({"Time": remap})

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(df4plot2, col="Growth Substrate",  row="Time", margin_titles=True, despine=False, sharey=True, sharex=True)
# g.map_dataframe(sns.boxplot, x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map, showfliers = False, order=["Pore throat", "Pore body"])
g.map_dataframe(sns.violinplot, 
                # x="Habitat Size", 
                x="Species", 
                y="Relative abundance", 
                # hue="Species", 
                hue="Habitat Size", 
                # palette = color_map, 
                # showfliers = False, 
                # order=["Pore throat", "Pore body"],
                order = ["Variovorax", "Sphingopyxis", "Rhodococcus"],
                )
# g.map_dataframe(sns.boxplot, x="Habitat Size", y="Relative abundance", hue="Species", palette=color_map, order=["(0.2~0.4)", "(0.4~0.6)", "(0.6~0.8)", "(0.8~1.0)"])
g.add_legend()
# plt.savefig(analysis_dir + "relative_abundance_vs_habitate_size_all_wo_outliers.pdf", bbox_inches='tight', dpi=300)


# %% [markdown]
# #### Box plots

# %%
import matplotlib.patches as mpatches

df = pop_den_1d_df.copy()
# df = df.loc[((df[["Variovorax", "Sphingopyxis", "Rhodococcus"]] != 0).all(axis=1)) & (df["Rep"] == "rep2")].copy()
df = df.loc[(df[["Variovorax", "Sphingopyxis", "Rhodococcus"]] != 0).all(axis=1)].copy()
df["Habitat Size"] = ""
df.loc[df["Relative Volume"] <= 0.4, "Habitat Size"] = "Pore throat"
df.loc[df["Relative Volume"] >= 0.8, "Habitat Size"] = "Pore body"
df.loc[(df["Relative Volume"] > 0.4) & (df["Relative Volume"] < 0.8), "Habitat Size"] = "Intermediate"

df4plot = pd.melt(df, id_vars=["Time", "Growth Substrate", "Habitat Size", "Radius", "Relative Volume"], value_vars=["Variovorax", "Sphingopyxis", "Rhodococcus"], var_name="Species", value_name="Relative abundance")

df4plot = df4plot.loc[(df4plot["Relative abundance"] > 0) & (df4plot["Habitat Size"] != "Intermediate")].copy()

matplotlib.rcParams['figure.figsize'] = (4, 4)

pairs=[
    [("Sphingopyxis", "Pore throat"), ("Sphingopyxis", "Pore body")], 
    [("Rhodococcus", "Pore throat"),("Rhodococcus", "Pore body")], 
    [("Variovorax", "Pore throat"), ("Variovorax", "Pore body")],
    ]

# Define orders and colors
species_order = ["Variovorax", "Sphingopyxis", "Rhodococcus"]
hue_order = ["Pore throat", "Pore body"]

# Colors: Light/Dark pairs for each species
# Variovorax (Red), Sphingopyxis (Green), Rhodococcus (Blue)
box_colors = [
    "#ffcccc", "#ccffcc", "#ccccff",   # Pore throat: Variovorax (light red), Sphingopyxis (light green), Rhodococcus (light blue)
    "#cc0000", "#008000", "#0000cc"    # Pore body: Variovorax (dark red), Sphingopyxis (dark green), Rhodococcus (dark blue)
]

# Build legend handles with correct colors
legend_handles = []
for si, sp in enumerate(species_order):
    for hi, h in enumerate(hue_order):
        color = box_colors[hi * len(species_order) + si]
        legend_handles.append(mpatches.Patch(facecolor=color, edgecolor='black', label=f"{sp} - {h}"))

for time in [11, 37, 90]:
    for carbon in ["N-acetylglucosamine", "Chitopentose", "Chitin"]:
        df4plot2 = df4plot.loc[(df4plot["Time"] == time) & (df4plot["Growth Substrate"] == carbon)].copy()

        # Skip empty subsets to avoid seaborn order/hue errors
        if df4plot2.empty:
            continue

        hue_plot_params = {
            'data': df4plot2,
            'x': 'Species',
            'y': 'Relative abundance',
            "order": species_order, 
            "hue": "Habitat Size",
            "hue_order": hue_order,
        }
        
        ax = sns.boxplot(data=df4plot2, 
                        x="Species", 
                        y="Relative abundance", 
                        hue="Habitat Size", 
                        showfliers = False, 
                        order = species_order,
                        hue_order=hue_order
                        )
        
        # Manually color the boxes
        boxes = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        if not boxes:
            boxes = [p for p in ax.artists if isinstance(p, mpatches.PathPatch)]
            
        for i, box in enumerate(boxes):
            if i < len(box_colors):
                box.set_facecolor(box_colors[i])

        annotator = Annotator(ax, pairs, **hue_plot_params)
        annotator.configure(test="Mann-Whitney").apply_and_annotate()
        # ax.legend(handles=legend_handles, fontsize='small', frameon=True)
        ax.legend_.remove()
        plt.savefig(analysis_dir + "relative_abundance_vs_habitate_size_" + carbon + "_at_" + str(time) + "_selective_wo_outliers.pdf", bbox_inches='tight', dpi=300)
        plt.close()


df4plot2 = df4plot.loc[(df4plot["Relative abundance"] > 0) & (df4plot["Time"].isin([11,37,90]))].copy()

plt.figure(figsize=(2, 2))
g = sns.FacetGrid(df4plot2, col="Growth Substrate",  row="Time", margin_titles=True, despine=False, sharey=True, sharex=True)

def colored_boxplot(data, **kwargs):
    ax = plt.gca()
    sns.boxplot(data=data, x="Species", y="Relative abundance", hue="Habitat Size", 
                order=species_order, hue_order=hue_order, showfliers=False, ax=ax)
    
    # Recolor boxes
    boxes = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
    if not boxes:
        boxes = [p for p in ax.artists if isinstance(p, mpatches.PathPatch)]
    for i, box in enumerate(boxes):
        if i < len(box_colors):
            box.set_facecolor(box_colors[i])
    ax.legend_.remove()

g.map_dataframe(colored_boxplot)
g.fig.legend(handles=legend_handles, fontsize='small', frameon=True, loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.savefig(analysis_dir + "relative_abundance_vs_habitate_size_all_wo_outliers.pdf", bbox_inches='tight', dpi=300)

# %%
