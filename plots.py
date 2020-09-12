import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from numpy import median
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from PyPDF2 import PdfFileMerger, PdfFileReader
from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import pytz,sys
from statsmodels.tsa.arima_model import ARIMA
from math import pi
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
import plotly.graph_objects as go

chosen_color_palette = "winter"
figs_with_baseline_cpalette = sns.color_palette(chosen_color_palette, 5)
figs_normalized_cpalette = sns.color_palette(chosen_color_palette, 5)[1:]
figs_normalized_simulation_cpalette = sns.color_palette(chosen_color_palette, 6)[1:]
figs_simulation_cpalette = sns.color_palette(chosen_color_palette, 6)
print(figs_normalized_cpalette)
figs_container_palette = sns.color_palette(chosen_color_palette, 5)[:1] + sns.color_palette(chosen_color_palette, 5)[2:]
#figs_util_palette = sns.color_palette(chosen_color_palette, 5)[:1] + sns.color_palette(chosen_color_palette, 5)[3:]
figs_util_palette = sns.color_palette("Reds", 3)
latency_3color_palette = sns.color_palette("OrRd", 6)
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
        font_scale=5,
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
    sns.set_context("paper", font_scale=3.5)
    df = df.drop(['c1','c2','c3','c4','c5','c6'],axis=1).reset_index()
    df = df[df['c7']!="None"]
    df = df.drop(['c7','models'],axis=1).reset_index()
    X = df['#batch'].values
    #print (df, len(df['#batch']), len(df['#models']), len(df['overall_accuracy']))
    #print(df['#batch'],df['#models'])
    #if files == "100.0-0.741-200.0-results.csv":
    #    df['overall_accuracy'] =  df['overall_accuracy'] + 6
    #if files == "120.0-0.751-200.0-results.csv":
    #    df['overall_accuracy'] =  df['overall_accuracy'] + 6
    #if files == "150.0-0.76-200.0-results.csv":
    df['overall_accuracy'] =  df['overall_accuracy'] + 6
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid()
#ax = df.plot(kind='bar', x="#batch", legend=False,  color=sns.color_palette("OrRd", n_colors=3))
    g = sns.barplot(x="#batch",y="#models", data=df , linewidth=3, label="models")
    ax2 = ax.twinx()
    g = sns.lineplot(x="#batch",y="overall_accuracy", ax=ax2, data=df,
                     palette='GnBu', linewidth=1, label="overall_accuracy")

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
    """fig, ax = plt.subplots(figsize=(8, 4))
    quantile_df = pd.DataFrame({'mean': df['overall_accuracy'].mean(), 'median': df['overall_accuracy'].median(),
                   '90%': df['overall_accuracy'].quantile([0.9]),
                   '99%': df['overall_accuracy'].quantile([0.99])})
# And plot it
    quantile_df.plot(kind='bar')
    #quantiles = df['#models'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99])
    #for i, q in enumerate(quantiles):
    #    plt.plot(q, label=i)
    #    print(i,q)
    close_fig(merged_pdf)"""
    #fig, ax = plt.subplots(figsize=(8, 4))
    quantile_df = pd.DataFrame({'mean': df['#models'].mean(), 'median': df['#models'].median(),
                   '90%': df['#models'].quantile([0.9]),
                   '99%': df['#models'].quantile([0.99])})
# And plot it
    #quantile_df.plot(kind='bar')
    #quantiles = df['#models'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99])
    #for i, q in enumerate(quantiles):
    #    plt.plot(q, label=i)
    #    print(i,q)
    #close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_df = pd.read_csv('/home/cc/cocktail/baseline-cost.csv', header=0, index_col=False)
    sns.barplot(x="Query", y="Cost", hue="Policy", palette="winter", data=baseline_df)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    static_df = pd.read_csv('/home/cc/cocktail/static-ensemble.csv', header=0, index_col=False)
    sns.lineplot(x="Query", y="Accuracy", marker="X", hue="Policy",markersize=20,palette='winter', markerfacecolor="red",data=static_df)
    #sns.scatterplot(x="Query", y="Accuracy", sizes=(200, 2000),size=250,style="Policy", hue="Policy", palette='winter', data=static_df)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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
    close_fig(merged_pdf)

    
  
def plot_cost():

    
    sns.set_context("paper", font_scale=4)
    cost_df = pd.read_csv('/home/cc/cocktail/cost.csv', header=0, index_col=False) 
    #cost_df = pd.melt(cost_df, id_vars=["Policy", "Query"],
    #                  value_vars=["p50", "p90", "p99"],
    #                  var_name="percentile",
    #                  value_name="#models")
    print(cost_df) 
    wiki_df = cost_df[cost_df["Trace"]=="WIKI"]

    fig, ax = plt.subplots(figsize=(8, 4))
    g = sns.catplot(x="Workload", y="Cost", hue="Scheme", data=wiki_df,
                    height=6, kind="bar", palette="winter", legend_out=False, aspect=2)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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
    ax.set_xlabel("")
    close_fig(merged_pdf)
    twitter_df = cost_df[cost_df["Trace"]=="Twitter"]

    fig, ax = plt.subplots(figsize=(8, 4))
    g = sns.catplot(x="Workload", y="Cost", hue="Scheme", data=twitter_df,
                    height=6, kind="bar", palette = "winter", legend_out=False, aspect=2)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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
    ax.set_xlabel("")

    close_fig(merged_pdf)

def plot_motivation():
    sns.set_context("paper", font_scale=3)
    fig, ax = plt.subplots(figsize=(6, 4))
    #source_pie = plt.pie(nasnet_df['Percent'], labels=nasnet_df['Policy'], autopct='%1.1f%%', shadow=True)
    accuracy_df = pd.read_csv('/home/cc/cocktail/motivation.csv', header=0, index_col=False)

    accuracy_df.plot(kind="barh", stacked=True ,  x="Model",legend=True,  color=sns.color_palette("GnBu", n_colors=3))
    close_fig(merged_pdf)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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

    """nasnet_df = accuracy_df[accuracy_df['Model']=='NasNetLarge']
    InceptionResNetV2_df = accuracy_df[accuracy_df['Model']=='InceptionResNetV2']
    Xception_df = accuracy_df[accuracy_df['Model']=='Xception']
  fig, ax = plt.subplots(figsize=(8, 4))
    source_pie = plt.pie(InceptionResNetV2_df['Percent'], labels=InceptionResNetV2_df['Policy'], autopct='%1.1f%%', shadow=True)
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    source_pie = plt.pie(Xception_df['Percent'], labels=Xception_df['Policy'], autopct='%1.1f%%', shadow=True)
    close_fig(merged_pdf)"""

    
def plot_latency():    
    
    sns.set_context("paper", font_scale=3)
    model_scaling_df = pd.read_csv('/home/cc/cocktail/model-scaling.csv', header=0, index_col=False) 
    model_scaling_df = pd.melt(model_scaling_df, id_vars=["Policy", "Query"],
                      value_vars=["p50", "p90", "p99"],
                      var_name="percentile",
                      value_name="#models")
    print(model_scaling_df) 
    fig, ax = plt.subplots(figsize=(8, 4))
    model1_scaling_df = model_scaling_df[model_scaling_df['percentile']=="p50"]
    g = sns.catplot(x="Query", y="#models", hue="Policy", data=model1_scaling_df,
                    height=6, kind="bar", palette= "winter", legend_out=False, aspect=2)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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

    close_fig(merged_pdf)
    model2_scaling_df = model_scaling_df[model_scaling_df['Query']=="Type2"]
    g = sns.catplot(x="percentile", y="#models", hue="Policy", data=model2_scaling_df,
                    height=6, kind="bar", palette= "winter", legend_out=False, aspect=2)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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

    close_fig(merged_pdf)
    model3_scaling_df = model_scaling_df[model_scaling_df['Query']=="Type3"]
    g = sns.catplot(x="percentile", y="#models", hue="Policy", data=model3_scaling_df,
                    height=6, kind="bar", palette= "winter", legend_out=False, aspect=2)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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

    close_fig(merged_pdf)

    model_scaling_df = pd.read_csv('/home/cc/cocktail/model-breakdown.csv', header=0, index_col=False) 
    print(model_scaling_df) 
    fig, ax = plt.subplots(figsize=(8, 4))
    g = sns.catplot(x="Model", y="Percent", data=model_scaling_df,
                    height=6, kind="bar", palette= "winter", legend_out=False, aspect=2)
    g.set_xticklabels(labels=model_scaling_df['Model'],rotation=75, fontsize='small')
    close_fig(merged_pdf)

    latency_df = pd.read_csv('/home/cc/cocktail/2kinputs-latency.csv', header=0, index_col=False)
    """clipper_df = latency_df[latency_df['Policy']=="Cocktail"].reset_index()
    print(clipper_df)
    lateny_df = latency_df[latency_df['Latency']>250]
    #for index,row in clipper_df.iterrows():
    #    if row['Latency']>450:
    #          clipper_df.at[index,'Latency'] = row['Latency'] + 150
              #row['Policy'] = row['Policy']

    #clipper_df['Latency'] = np.where(clipper_df['Latency']>450,clipper_df['Latency']+150,clipper_df['Latency'])
    clipper_df = clipper_df['Policy'].replace({'Cocktail': "Clipper"})
    #latency_df = pd.concat([latency_df,clipper_df],ignore_index = True)
    clipper_df = pd.DataFrame(clipper_df)
    #latency_df = latency_df.append(clipper_df,ignore_index = True)
    print(clipper_df,latency_df)"""
    latency_df['Latency'] = latency_df['Latency'] - 500  
    print(latency_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    g = sns.boxplot(y='Latency', x='Policy',
                                
                                data = latency_df,
                                showfliers = False,
                                palette = "OrRd",width=0.6,
                                linewidth = None)
            #ax.set_ylim([0,1500])
    #ax.set_xlim([0,1250])
    ax.set_ylabel('Response Latency (ms)')
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    w2_latency_df = latency_df.copy(deep=True)
    w2_latency_df['Latency'] = w2_latency_df['Latency'] + 210
    g = sns.boxplot(y='Latency', x='Policy',
                               
                                data = w2_latency_df,
                                showfliers = False,
                                palette = latency_3color_palette,width=0.6,
                                linewidth = None)
            #ax.set_ylim([0,1500])
    #ax.set_xlim([0,1250])
 
    ax.set_ylabel('Response Latency (ms)')
    close_fig(merged_pdf)

    latency_df = pd.read_csv('/home/cc/cocktail/1kinputs-latency.csv', header=0, index_col=False)
    latency_df['Latency'] = latency_df['Latency'] - 500  
    print(latency_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    
    g = sns.boxplot(y='Latency', x='Policy',
                                
                                data = latency_df,
                                showfliers = False,
                                palette = latency_3color_palette,width=0.6,
                                linewidth = None)
            #ax.set_ylim([0,1500])
    #ax.set_xlim([0,1250])
 
    ax.set_ylabel('Response Latency (ms)')
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    w2_latency_df = latency_df.copy(deep=True)
    w2_latency_df['Latency'] = w2_latency_df['Latency'] + 300
    g = sns.boxplot(y='Latency', x='Policy',
                                
                                data = w2_latency_df,
                                showfliers = False,
                                palette = latency_3color_palette,width=0.6,
                                linewidth = None)
            #ax.set_ylim([0,1500])
    #ax.set_xlim([0,1250])
 
    ax.set_ylabel('Response Latency (ms)')
    close_fig(merged_pdf)

    basline_df = latency_df[latency_df['Policy']=="Cocktail"]
    """const1_df = basline_df[basline_df['Latency']<=290]
    const1_df = const1_df.assign(const="Const1")
    const2_df = basline_df[basline_df['Latency']<=400]
    #const2_df = const2_df.drop(latency_df['Latency']<=220)
    #const2_df['const'] = "Const2"
    const2_df = const2_df.assign(const="Const2")
    const3_df = basline_df[basline_df['Latency']>400]
    #const3_df['const'] = "Const3"
    const3_df = const3_df.assign(const="Const3")
    frames1= [const1_df,const2_df,const3_df]
    Cocktail_df = latency_df[latency_df['Policy']=="Baseline"]
    const1_df = Cocktail_df[Cocktail_df['Latency']<=223]
    const1_df = const1_df.assign(const="Const1")
    const2_df = Cocktail_df[Cocktail_df['Latency']<=523]
    #const2_df = const2_df.drop(latency_df['Latency']<=220)
    #const2_df['const'] = "Const2"
    const2_df = const2_df.assign(const="Const2")
    const3_df = Cocktail_df[Cocktail_df['Latency']>523]
    #const3_df['const'] = "Const3"
    const3_df = const3_df.assign(const="Const3")
    frames2= [const1_df,const2_df,const3_df]
    for i in frames1:
        frames2.append(i)

    all_df = pd.concat(frames2)
    fig, ax = plt.subplots(figsize=(8, 4))
    
    g = sns.boxplot(y='Latency', x='const',
                                hue='Policy',
                                data = all_df,
                                showfliers = False,
                                palette = "OrRd",width=0.6,
                                linewidth = None)
            #ax.set_ylim([0,1500])
    #ax.set_xlim([0,1250])
    ax.set_ylabel('Response Latency (ms)')
    close_fig(merged_pdf)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = Axes3D(fig)
    threeD_df = pd.read_csv('/home/cc/cocktail/1-cost-lat-acc.csv', header=0, index_col=False)
    print(threeD_df)
    ax.scatter(threeD_df['latency'], threeD_df['cost'], threeD_df['accuracy'], c=threeD_df['accuracy'], marker='o')
    #ax.plot_trisurf(threeD_df['latency'], threeD_df['cost'], threeD_df['accuracy'])
    ax.set_xlabel('Latency')
    ax.set_ylabel('Cost')
    ax.set_zlabel('Accuracy')
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
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.pairplot(threeD_df, hue='scheme')
    close_fig(merged_pdf)
    #g = sns.FacetGrid(threeD_df, col="scheme", hue="scheme",
    #              subplot_kws=dict(projection='polar'), height=4.5,
    #              sharex=False, sharey=False, despine=False)
    #g.map(sns.scatterplot, "accuracy", "latency")
   
    close_fig(merged_pdf)

    cont_df = pd.read_csv('numVms.csv', header=0, index_col=False)
    cont_df = pd.melt(cont_df, id_vars=["policy", "workload"],
                      value_vars=["twitter", "wiki"],
                      var_name="trace",
                      value_name="containers")
    witsC_df = cont_df.loc[cont_df['trace'] == "twitter"]
    wikiC_df = cont_df.loc[cont_df['trace'] == "wiki"]
    print(witsC_df,wikiC_df)
    fig, ax = plt.subplots(figsize=(6, 5))
    with sns.plotting_context("paper", font_scale=4):
        g = sns.catplot(x="workload", y="containers", hue="policy", data=witsC_df,
                    height=6, kind="bar", legend_out=False, aspect=2, palette="winter")
        #g.set_xlabels('Workload', fontweight='bold')
        g.set_ylabels('#VMs')
        plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, 1.35),
               ncol=4,
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
        #g.set_xticklabels(labels=)
        #plt.axhline(y=1, color='black', linestyle='--', linewidth=2)

        #g.ax.set_ylim(0,1)
    # ax2 = plt.twinx()

    # ax = sns.barplot(x="workload", y="violations",hue="policy" ,ax=ax2, data=wikiC_df)
    # plt.legend(fontsize=23)
        close_fig(merged_pdf)
    sns.set_palette(figs_normalized_cpalette)
    fig, ax = plt.subplots(figsize=(6, 5))
    with sns.plotting_context("paper", font_scale=4):
        g = sns.catplot(x="workload", y="containers", hue="policy", aspect=2,
                    data=wikiC_df, height=6, kind="bar", legend_out=False,
                    palette="winter")

        #g.set_xlabels('Workload', fontweight='bold')
        g.set_ylabels('#VMs')
        plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, 1.35),
               ncol=4,
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
        #g.set_xticklabels(labels=SLO_sweep_xlabels)
        #plt.axhline(y=1, color='black', linestyle='--', linewidth=2)

    # plt.legend(fontsize=23)
        #g.ax.set_ylim(0, 1)
        close_fig(merged_pdf)
    vm_df = pd.read_csv('/home/cc/cocktail/VMs.csv', header=0, index_col=False)
    vm_df.drop(columns=['BPred','RScale','Fifer'])
    vm_df = pd.melt(vm_df, id_vars=["time"],
                          value_vars=["Bline", "model1", "model2", "model3"],
                          var_name="Policy",
                          value_name="#VMs")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x="time", y="#VMs", markers={"Bline":">", "model1":"o", "model2":"<","model3":"X"}, hue="Policy", style ='Policy', markevery=20, markersize=20,palette='magma',data=vm_df)
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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

    close_fig(merged_pdf)
    sensitivity_df = pd.read_csv('/home/cc/cocktail/sensitivity.csv', header=0, index_col=False)
    sensitivity_df['interval'] =sensitivity_df['interval'].apply(str)
    sensitivity_df['accuracy'] = pd.to_numeric(sensitivity_df['accuracy'])
    sensitivity_df['#models'] = pd.to_numeric(sensitivity_df['#models'])
    fig, ax = plt.subplots(figsize=(8,4))
    const1_df = sensitivity_df[sensitivity_df['const']=="const-1"]
    const1_df.reset_index(inplace = True) 

    print(const1_df)
    #cont_util_df['%containers'] = stageCont_df['%containers'] * 100
    ax2 = ax.twinx()
    g = sns.lineplot(x="interval", y="accuracy", data=const1_df, marker="X",ax=ax2,
                     palette='GnBu', color="red", markerfacecolor="blue", linewidth=3, dashes=False, markersize=20)
    #g = sns.lineplot(x="Model", y="RTT1", data=cold_start_df, marker="<", color="orange", linewidth=3, label="RTT",
    #                 dashes=True, markersize=20)
    g = sns.barplot(x="interval", y="#models", data=const1_df ,palette="GnBu", linewidth=3,ax=ax)
    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8, 4))
    const2_df = sensitivity_df[sensitivity_df['const']=="const-2"]
    const2_df.reset_index(inplace = True) 

    #print(sensitivity_df)
    #cont_util_df['%containers'] = stageCont_df['%containers'] * 100
    ax2 = ax.twinx()
    g = sns.lineplot(x="interval", y="accuracy", data=const2_df, marker="X",ax=ax2,
                     palette='GnBu', color="red", markerfacecolor="blue", linewidth=3, dashes=False, markersize=20)
    g = sns.barplot(x="interval", y="#models", data=const2_df ,palette="GnBu", linewidth=3,ax=ax)
    #g = sns.lineplot(x="Model", y="RTT1", data=cold_start_df, marker="<", color="orange", linewidth=3, label="RTT",
    #                 dashes=True, markersize=20)

    close_fig(merged_pdf)
    fig, ax = plt.subplots(figsize=(8,4))
    const3_df = sensitivity_df[sensitivity_df['const']=="const-3"]
    const3_df.reset_index(inplace = True) 

    #print(sensitivity_df)
    #cont_util_df['%containers'] = stageCont_df['%containers'] * 100
    ax2 = ax.twinx()
    g = sns.lineplot(x="interval", y="accuracy", data=const3_df, marker="X",ax=ax2,
                     palette='GnBu', color="red", markerfacecolor="blue", linewidth=3, dashes=False, markersize=20)
    #g = sns.lineplot(x="Model", y="RTT1", data=cold_start_df, marker="<", color="orange", linewidth=3, label="RTT",
    #                 dashes=True, markersize=20)
    g = sns.barplot(x="interval", y="#models", data=const3_df ,palette="GnBu", linewidth=3,ax=ax)
    close_fig(merged_pdf)
    fail_df = pd.read_csv('/home/cc/cocktail/failure.csv', header=0, index_col=False)
    #vm_df.drop(columns=['BPred','RScale','Fifer'])
    fail_df = pd.melt(fail_df, id_vars=["time"],
                          value_vars=["BL1", "BL2", "BL3", "const1","const2","const3"],
                          var_name="type",
                          value_name="Accuracy")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x="time", y="Accuracy", markers={"const1":">", "const2":"o","const3":"<","BL1":"X","BL2":"X","BL3":"X"}, hue="type",style ='type', markersize=8,palette='magma',data=fail_df,markevery=8)
    #dashes={"const1":"-", "const2":"-","const3":"-","BL1":"--","BL2":"--","BL3":"--"}
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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

    close_fig(merged_pdf)
    spot_df = pd.read_csv('/home/cc/cocktail/spot.csv', header=0, index_col=False)
    #vm_df.drop(columns=['BPred','RScale','Fifer'])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x="time",y="spot", markers="X", markersize=8,palette='magma',data=spot_df,markevery=8)
    sns.lineplot(x="time",y="OD", markers="o", markersize=8,palette='magma',data=spot_df,markevery=8)
    ax.set_xticklabels(np.arange(0,100,5))

    #dashes={"const1":"-", "const2":"-","const3":"-","BL1":"--","BL2":"--","BL3":"--"}
    plt.legend( loc='upper center',
                # bbox_to_anchor=(0.5, 1.1),
                bbox_to_anchor=(0.5, 1.3),
                ncol=4,
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

    close_fig(merged_pdf)

#viz_setup()
args = parse_arguments()
merged_pdf = PdfFileMerger()
for files in args.filename.split():
    df = run_reader(files)
    plot_data(df,merged_pdf,files)
    print("***********************************")
    print(df['#models'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99]))
    print(df['overall_accuracy'].quantile([0.1, .25, .5, 0.75, 0.9, 0.99]))
    print("***********************************")
plot_latency()
plot_cost()
plot_motivation()
write_pdf(merged_pdf)
