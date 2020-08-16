import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from numpy import median
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from PyPDF2 import PdfFileMerger, PdfFileReader
import datetime as dt
import pytz,sys
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import argparse
import ast
import csv
import glob
import io
import math
import os
import pprint
import sys
import matplotlib.ticker as ticker

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
def parse_arguments():
    try:
        args_parser = argparse.ArgumentParser(description="Analyze output logs")
        args_parser.add_argument('-i', "--input_csv", default='', action='store', dest='filename',help="Path of input CSV file with all container's information in CSV format")
        args_parser.add_argument('-f', "--output", default='', action='store', dest='output',help="Path of input CSV file with all container's information in CSV format")
        args = args_parser.parse_args()
    except:
        class ArgsDefault:
            filename = "na"
        args = ArgsDefault() 
        args.filename="first-working-logs.csv"
        args.output = "figs.pdf"
    return args
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
def run_reader(csv_file):
    print ("Opening file: {}".format(csv_file))
    #cols=['asr-runtime','asr-queue','nlp-runtime','nlp-queue','qa-runtime','qa-queue','policy']
    cols=['c1','#batch','c2','overall_accuracy','c3','step_accuracy','c4','models','c5','#models','c6','c7']
    df = pd.read_csv(csv_file, header=None, names=cols,sep=",", index_col=False)
    #print(getframeinfo(currentframe()).lineno,df)
    return df

def close_fig(mypdf):
    buf=io.BytesIO()
    plt.savefig(buf, format='pdf', dpi=400,bbox_inches='tight')
    buf.seek(0)
    mypdf.append(PdfFileReader(buf))
    plt.cla()
    plt.clf()
    plt.close('all')

def write_pdf(mypdf):
    global args
    print("args.output", args.output)
    mypdf.write(args.output)
    print("pdf written")


def plot_data(df, merged_pdf,files):
    df = df.drop(['c1','c2','c3','c4','c5','c6'],axis=1).reset_index()
    df = df[df['c7']!="None"]
    df = df.drop(['c7','models'],axis=1).reset_index()
    X = df['#batch'].values
    #print (df, len(df['#batch']), len(df['#models']), len(df['overall_accuracy']))
    #print(df['#batch'],df['#models'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid()
#ax = df.plot(kind='bar', x="#batch", legend=False,  color=sns.color_palette("OrRd", n_colors=3))

    g = sns.barplot(x="#batch",y="#models", data=df , linewidth=3, label="models").set_title(files.strip("csv"))
    ax2 = ax.twinx()
    g = sns.lineplot(x="#batch",y="overall_accuracy", ax=ax2, data=df,
                     palette='Paired', color="red", linewidth=1, label="overall_accuracy")

    plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, 1.4),
               ncol=2,
               frameon=True,
               fancybox=True,
               framealpha=0.4,
               shadow=False,
               edgecolor="black",
               labelspacing=0,
               columnspacing=0.1,
               handletextpad=0.5
               # labels=figs_normalized_barlabels
               )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xticklabels(df['#batch'])
    ax.set_xlim(0,len(df['#models']))
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    quantile_df = pd.DataFrame({'mean': df['overall_accuracy'].mean(), 'median': df['overall_accuracy'].median(),
                   '90%': df['overall_accuracy'].quantile([0.9]),
                   '99%': df['overall_accuracy'].quantile([0.99])})
# And plot it
    quantile_df.plot(kind='bar')
    #quantiles = df['#models'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99])
    #for i, q in enumerate(quantiles):
    #    plt.plot(q, label=i)
    #    print(i,q)
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    quantile_df = pd.DataFrame({'mean': df['#models'].mean(), 'median': df['#models'].median(),
                   '90%': df['#models'].quantile([0.9]),
                   '99%': df['#models'].quantile([0.99])})
# And plot it
    quantile_df.plot(kind='bar')
    #quantiles = df['#models'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99])
    #for i, q in enumerate(quantiles):
    #    plt.plot(q, label=i)
    #    print(i,q)
    close_fig(merged_pdf)



args = parse_arguments()
#viz_setup()
merged_pdf = PdfFileMerger()
for files in args.filename.split():
    df = run_reader(files)
    plot_data(df,merged_pdf,files)
write_pdf(merged_pdf)
print("***********************************")
print(df['#models'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99]))
print(df['overall_accuracy'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99]))
print("***********************************")
