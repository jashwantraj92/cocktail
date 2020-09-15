#%%
#!/bin/env python3

import argparse
import ast
import csv
import glob
import io
import math
import os
import pprint
import sys
from collections import defaultdict
import itertools as it

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

import scipy.stats as stats
import numpy as np
from numpy import median
import pandas as pd
import seaborn as sns
from PyPDF2 import PdfFileMerger, PdfFileReader
import datetime as dt
import pytz
from inspect import currentframe, getframeinfo
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from math import pi
#%%

chosen_color_palette = "GnBu"
figs_with_baseline_cpalette = sns.color_palette(chosen_color_palette, 5)
figs_normalized_cpalette = sns.color_palette(chosen_color_palette, 5)[1:]
figs_normalized_simulation_cpalette = sns.color_palette(chosen_color_palette, 6)[1:]
figs_simulation_cpalette = sns.color_palette(chosen_color_palette, 6)
print(figs_normalized_cpalette)
figs_container_palette = sns.color_palette(chosen_color_palette, 5)[:1] + sns.color_palette(chosen_color_palette, 5)[2:]
#figs_util_palette = sns.color_palette(chosen_color_palette, 5)[:1] + sns.color_palette(chosen_color_palette, 5)[3:]
figs_util_palette = sns.color_palette("Reds", 3)
latency_3color_palette = sns.color_palette("GnBu", 6)
tail_color_palette = sns.color_palette("GnBu", 6)[3:6]
#boxplot_3color_palette = sns.light_palette("YlGnBu", 3)

figs_with_baseline_barlabels = ['Bline','S_Batch.','RScale','Fifer']
figs_normalized_barlabels= ['Static Batching.','RScale','Fifer']
SLO_sweep_xlabels = ['Heavy','Medium','Light']
merged_pdf = PdfFileMerger()
def viz_setup():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")    
  
    # Set the font to be serif, rather than sans
    custom_rc={
        'grid.linestyle' : '--',
        'font.family': ['sans-serif'],
        'font.sans-serif': [
            'DejaVu Serif',
            'sans-serif',
            'Arial'],
        'axes.grid':True,
    }

    # Enable ticks
    sns.set(
        style='ticks',
        font_scale=3,
        color_codes=False,
        rc=custom_rc, 
    )

    #Control borders around chart.
    # sns.despine()
    sns.despine(offset=10, trim=True)

def close_fig(mypdf):
    buf=io.BytesIO()
    plt.savefig(buf, format='pdf', dpi=400,bbox_inches='tight')
    buf.seek(0)
    mypdf.append(PdfFileReader(buf))
    plt.cla()
    plt.clf()
    plt.close('all')

def write_pdf(mypdf):
    mypdf.write("figs.pdf")
    #cold_start_df = pd.melt(cold_start_df, id_vars=["policy"],
    #                       value_vars=["stage1", "stage2", "stage3"],
    #                       var_name="Stage",
    #                       value_name="#Jobs")
    #print(getframeinfo(currentframe()).lineno,stageCont_df)

def plot():
    global merged_pdf
    cold_start_df = pd.read_csv('cold-starts.csv', header=0, index_col=False)
    #cold_start_df["cold_execution"] = pd.to_numeric(cold_start_df["cold_execution"])
    #cold_start_df["warm_execution"] = pd.to_numeric(cold_start_df["warm_execution"])
    cs_df = cold_start_df[["Model", "cold_execution", "RTT1"]]
    cs_df.set_index('Model', inplace=True)
    print(cold_start_df,cs_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    #cont_util_df['%containers'] = stageCont_df['%containers'] * 100
    g = sns.lineplot(x="Model", y="cold_execution", data=cold_start_df, marker="X",
                     palette='Paired', color="red", markerfacecolor="blue", linewidth=3, label="exec_time", dashes=False, markersize=20,sort=False)
    #g = sns.lineplot(x="Model", y="RTT1", data=cold_start_df, marker="<", color="orange", linewidth=3, label="RTT",
    #                 dashes=True, markersize=20)
    g = sns.barplot(x="Model", y="RTT1", data=cold_start_df ,palette="Blues", linewidth=3, label="RTT")
    """ax = sns.lineplot(  data=cs_df, 
                        dashes=True, 
                        palette="bright", 
                        legend="full",
                        # markers=True,
                        markers=["o","s"],
                        markersize=20,
                        linewidth=3,
                    )"""
    plt.legend(loc='upper center',
                bbox_to_anchor=(0.5, 1.2),
                ncol=2,
                frameon=False,
                fancybox=False,
                framealpha=0.4,
                shadow=False,
                edgecolor="black",
                # labels=figs_normalized_barlabels
                )
    g.set_ylabel("Latency (ms)")
    g.set_xlabel('Model',fontweight="bold")
    plt.xticks(rotation=75, ha='right')
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(10, 6))
    # cont_util_df['%containers'] = stageCont_df['%containers'] * 100
    g = sns.lineplot(x="Model", y="warm_execution", data=cold_start_df, marker="X",
                     palette='Paired', linewidth=3,color="red", markerfacecolor="blue", label="exec_time", dashes=False, markersize=20,sort=False)
    g = sns.barplot(x="Model", y="RTT2", data=cold_start_df ,palette="Blues", linewidth=3, label="RTT")
    #g = sns.lineplot(x="Model", y="RTT2", data=cold_start_df, marker="<", color="orange", linewidth=3, label="RTT", dashes=True, markersize=20)
    plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, 1.2),
               ncol=2,
               frameon=False,
               fancybox=False,
               framealpha=0.4,
               shadow=False,
               edgecolor="black",
               # labels=figs_normalized_barlabels
               )
    # g.set_xticklabels(labels=SLO_sweep_xlabels)
    g.set_ylabel("Latency (ms)")
    g.set_xlabel('Model',fontweight="bold")
    plt.xticks(rotation=75, ha='right')
    close_fig(merged_pdf) 
viz_setup()
plot()


df = pd.DataFrame({
'Model': ['InFaaS','Clipper','Cocktail'],
'Cost': [0.36,0.89,0.26],
'Latency': [0.837,0.3931,0.38],
'Accuracy-loss': [0.8,0.53,0.26]
})
 
 
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
fig, ax = plt.subplots(figsize=(8, 4))

ax = plt.subplot(111, polar=True)

 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='black', size=23)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2,0.4,0.6,0.8],["0.2","0.4","0.6","0.8"], color="black", size=20)
#plt.ylim(0,40)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('Model').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', color="blue",label="InFaas")
ax.fill_between(angles, values, 'b', color="blue", alpha=0.1)
 
# Ind2
values=df.loc[1].drop('Model').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid',color="red", label="Clipper")
ax.fill(angles, values, 'r', color="red",alpha=0.1)

values=df.loc[2].drop('Model').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', color="green",label="Cocktail")
ax.fill(angles, values, 'y', alpha=0.1,color="green")
plt.legend( loc='lower center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, -0.35),
                ncol=3,
                frameon=True,
                fancybox=True, 
                framealpha=0.4,
                shadow=False,
                edgecolor="black",
                labelspacing=0,
                columnspacing=0.1,
                handletextpad=0.5,
                fontsize=23
                # labels=figs_normalized_barlabels
                )

"""
#plt.title(title, size=11, color=color, y=1.1)

# ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)

# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df.index))

# Loop to plot
for row in range(0, len(df.index)):
    make_spider( row=row, title='group '+df['group'][row], color=my_palette(row))
"""
my_palette = plt.cm.get_cmap("Set2", len(df.index))

#plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol = 2, fontsize= 'x-large')
close_fig(merged_pdf)
write_pdf(merged_pdf)
