"""
Visualization script

Written by Ed Oughton

January 2020

"""
import os
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import contextily as ctx
import geopandas as gpd

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..','scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')

def create_scatterplot():

    clust_averages = pd.read_csv(os.path.join(DATA_PROCESSED,
        'lsms-cluster-2016.csv'))

    to_plot = clust_averages[["nightlights", "cons", "urban"]]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    g = sns.scatterplot(x="nightlights", y="cons", hue="urban", data=to_plot, ax=ax1)
    g.set(xlabel='Luminosity (DN)', ylabel='Household Consumption ($ per day)',
        title='Consumption \n vs Luminosity')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=reversed(handles[1:3]), labels=reversed(labels[1:3]))

    to_plot = clust_averages[["cons_pred", "cons", "urban"]]

    g = sns.scatterplot(x="cons_pred", y="cons", hue="urban", data=to_plot, ax=ax2)
    g.set(xlabel='Predicted Consumption ($ per day)',
        ylabel='Household Consumption ($ per day)',
        title='Consumption \n vs Predicted Consumption')

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=reversed(handles[1:3]), labels=reversed(labels[1:3]))

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures','scatterplot.png'))

    return print('Completed scatterplot')


def create_regplot():

    clust_averages = pd.read_csv(os.path.join(DATA_PROCESSED,
        'lsms-cluster-2016.csv'))

    to_plot = clust_averages[["nightlights", "cons", "cons_pred", "urban"]]

    urban = to_plot.loc[to_plot['urban'] == 'Urban']
    rural = to_plot.loc[to_plot['urban'] == 'Rural']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,8))

    g = sns.regplot(x="nightlights", y="cons", data=urban, ax=ax1)
    g.set(xlabel='Luminosity (DN)', ylabel='Urban HH Consumption ($ per day)',
        title='Urban Consumption \n vs Luminosity')

    g = sns.regplot(x="nightlights", y="cons", data=rural, ax=ax2)
    g.set(xlabel='Luminosity (DN)', ylabel='Rural HH Consumption ($ per day)',
        title='Rural Consumption \n vs Luminosity')

    g = sns.regplot(x="cons_pred", y="cons", data=urban, ax=ax3)
    g.set(xlabel='Predicted Consumption ($ per day)',
        ylabel='Urban HH Consumption ($ per day)',
        title='Urban Consumption \n vs Predicted Consumption')

    g = sns.regplot(x="cons_pred", y="cons", data=rural, ax=ax4)
    g.set(xlabel='Predicted Consumption ($ per day)',
        ylabel='Rural HH Consumption ($ per day)',
        title='Rural Consumption \n vs Predicted Consumption')

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures','regplot.png'))

    return print('Completed regplot')


def plot_map(metric, legend_label, title, hour, roads, flow_min, flow_max, sites,
    output_filename, metric_min, metric_max):

    fig, ax = plt.subplots(figsize=(8, 10))

    plt.rcParams['savefig.pad_inches'] = 0
    plt.autoscale(tight=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #         hspace = -5, wspace = 0)

    roads.plot(
        # figsize=(8, 10),
        column=metric,
        cmap='RdYlBu',#'hot',#
        norm=matplotlib.colors.Normalize(vmin=-150, vmax=150),
        legend=True,
        legend_kwds={'label': legend_label, 'orientation': 'horizontal'},
        ax=ax
    )

    sites.plot(
        column = 'Cell Site',
        markersize=10,
        legend=True,
        ax=ax
        )

    # plt.legend(, bbox_transform=ax.transAxes)
    plt.title('{:02d}:00 {}'.format(hour, title), fontsize=16)
    ctx.add_basemap(ax, crs=roads.crs)
    plt.savefig(output_filename, pad_inches=0, bbox_inches='tight')
    plt.close()

    return print('Completed {:02d}.png'.format(hour))

if __name__ == '__main__':

    create_scatterplot()

    create_regplot()
