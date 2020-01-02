import matplotlib.pyplot as plt
import seaborn as sns

to_plot = clust_averages[["nightlights", "cons", "urban"]]

fig, (ax1, ax2) = plt.subplots(1, 2)
g = sns.scatterplot(x="nightlights", y="cons", hue="urban", data=to_plot, ax=ax1)
g.set(xlabel='Luminosity (DN)', ylabel='Household Consumption ($ per day)',
    title='Luminosity \n vs Consumption')

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=reversed(handles[1:3]), labels=reversed(labels[1:3]))

to_plot = clust_averages[["cons_pred", "cons", "urban"]]

g = sns.scatterplot(x="cons_pred", y="cons", hue="urban", data=to_plot, ax=ax2)
g.set(xlabel='Predicted Consumption ($ per day)',
    ylabel='Household Consumption ($ per day)',
    title='Predicted Consumption \n vs Consumption')

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=reversed(handles[1:3]), labels=reversed(labels[1:3]))

fig.tight_layout()
fig.savefig(os.path.join(BASE_PATH, '..', 'vis','figures','combined.png'))
