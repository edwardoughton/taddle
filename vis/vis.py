import os
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..','scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')

print('Writing all other data')
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
fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures','panel_plot.png'))
