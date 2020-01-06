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
import contextily as ctx
import geopandas as gpd
import matplotlib.colors

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..','scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results')

def create_scatterplot(metric, metric2, coefficients, distance):

    coeff_urban = round(coefficients['urban_{}'.format(distance)].values[0], 3)
    coeff_rural = round(coefficients['rural_{}'.format(distance)].values[0], 3)

    clust_averages = pd.read_csv(os.path.join(DATA_PROCESSED,
        'lsms-cluster-2016.csv'))

    to_plot = clust_averages[[metric, "cons", "urban", "population_{}".format(distance)]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    g = sns.scatterplot(x=metric, y="cons", hue="urban",
        size="population_{}".format(distance),
        data=to_plot, ax=ax1)
    g.set(xlabel='Luminosity (DN)', ylabel='Household Consumption ($ per month)',
        title='Consumption \n vs Luminosity ({})'.format(distance))

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(title='Settlement Type', handles=reversed(handles[1:3]),
        labels=['Rural ({})'.format(coeff_rural), 'Urban ({})'.format(coeff_urban)])

    to_plot = clust_averages[[metric2, "cons", "urban", "population_{}".format(distance)]]

    g = sns.scatterplot(x=metric2, y="cons", hue="urban",
        size="population_{}".format(distance), data=to_plot, ax=ax2)
    g.set(xlabel='Predicted Consumption ($ per month)',
        ylabel='Household Consumption ($ per month)',
        title='Consumption \n vs Predicted Consumption ({})'.format(distance))

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(title='Settlement Type', handles=reversed(handles[1:3]),
        labels=['Rural ({})'.format(coeff_rural), 'Urban ({})'.format(coeff_urban)])

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures',
                'scatterplot_{}.png'.format(distance)))

    return print('Completed scatterplot')


def create_regplot(coefficients, distance):

    coeff_urban = round(coefficients['urban_{}'.format(distance)].values[0], 3)
    coeff_rural = round(coefficients['rural_{}'.format(distance)].values[0], 3)

    clust_averages = pd.read_csv(os.path.join(DATA_PROCESSED,
        'lsms-cluster-2016.csv'))

    metric = "luminosity_sum_{}".format(distance)
    metric2 = 'cons_pred_{}'.format(distance)
    to_plot = clust_averages[[metric, "cons", metric2, "urban"]]

    urban = to_plot.loc[to_plot['urban'] == 'urban']
    rural = to_plot.loc[to_plot['urban'] == 'rural']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,8))

    g = sns.regplot(x=metric, y="cons", data=urban, ax=ax1)
    g.set(xlabel='Luminosity (DN)', ylabel='Urban HH Consumption ($ per day)',
        title='Urban Consumption ({})\n vs Luminosity ({})'.format(coeff_urban, distance))

    g = sns.regplot(x=metric, y="cons", data=rural, ax=ax2)
    g.set(xlabel='Luminosity (DN)', ylabel='Rural HH Consumption ($ per day)',
        title='Rural Consumption ({})\n vs Luminosity ({})'.format(coeff_rural, distance))

    g = sns.regplot(x=metric2, y="cons", data=urban, ax=ax3)
    g.set(xlabel='Predicted Consumption ($ per day)',
        ylabel='Urban HH Consumption ($ per day)',
        title='Urban Consumption ({})\n vs Predicted Consumption ({})'.format(coeff_urban, distance))

    g = sns.regplot(x=metric2, y="cons", data=rural, ax=ax4)
    g.set(xlabel='Predicted Consumption ($ per day)',
        ylabel='Rural HH Consumption ($ per day)',
        title='Rural Consumption ({})\n vs Predicted Consumption ({})'.format(coeff_rural, distance))

    fig.tight_layout()
    fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures',
                'regplot_{}.png'.format(distance)))

    return print('Completed regplot')


def plot_map(data, title, legend_label):

    fig, ax = plt.subplots(figsize=(8, 10))

    plt.rcParams['savefig.pad_inches'] = 0
    plt.autoscale(tight=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    data.plot(
        column='pred_consu',
        cmap='RdYlBu',
        norm=matplotlib.colors.Normalize(vmin=0, vmax=200),
        legend=True,
        legend_kwds={'label': legend_label, 'orientation': 'horizontal'},
        alpha=0.3,
        ax=ax
    )

    plt.title(title, fontsize=16)
    ctx.add_basemap(ax, crs=data.crs)
    plt.savefig(os.path.join(BASE_PATH, '..', 'vis','figures',
                'context.png'), pad_inches=0, bbox_inches='tight')
    plt.close()

    return print('Completed')

if __name__ == '__main__':

    coefficients = pd.read_csv(os.path.join(DATA_RESULTS, 'coefficients.csv'))

    create_scatterplot("luminosity_sum_1km", "cons_pred_1km", coefficients, '1km')
    create_scatterplot("luminosity_sum_10km", "cons_pred_10km", coefficients, '10km')

    create_regplot(coefficients, '1km')
    create_regplot(coefficients, '10km')

    results = gpd.read_file(os.path.join(DATA_RESULTS, 'results.shp'))

    plot_map(results, 'Predicted Monthy Consumption ($)', 'Monthy Consumption ($)')
