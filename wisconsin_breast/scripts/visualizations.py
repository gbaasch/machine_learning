import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
import numpy as np

train = pd.read_csv("../train.csv")


def plot_feature_histograms():
    print "=========== plotting histograms"
    for x in train:
        if train[x].dtype == np.float64 or train[x].dtype == np.int64:
            ax = train[x].plot.hist(title=x)
            fig = ax.get_figure()
            fig.savefig('../visualizations/histograms/{}.jpeg'.format(x))
            fig.clear()
    print "=========== done"


def plot_correlations():
    print "=========== plotting correlation"
    ax = plt.matshow(train.corr())
    fig = ax.get_figure()
    fig.savefig('../visualizations/correlation/heatmap.jpeg')
    fig.clear()
    print "=========== done"


def main():
    print(train.head())
    plot_feature_histograms()
    plot_correlations()


main()
