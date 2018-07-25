import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

train = pd.read_csv("../train.csv")


def plot_feature_histograms(directory):
    print "=========== plotting histograms"
    for x in train:
        if train[x].dtype == np.float64 or train[x].dtype == np.int64:
            ax = train[x].plot.hist(title=x)
            fig = ax.get_figure()
            fig.savefig('{}/{}.jpeg'.format(directory, x))
            fig.clear()
    print "=========== done"


def plot_correlations(directory):
    print "=========== plotting correlation"
    ax = plt.matshow(train.corr())
    fig = ax.get_figure()
    fig.savefig('{}/correlation/heatmap.jpeg'.format(directory))
    fig.clear()
    print "=========== done"


def plot_features_against_residuals(X_dev_example, labels, predictions,
                                    directory):
    print "=========== plotting features against residuals"
    fig = plt.figure()
    for i in range(0, len(X_dev_example[0])):
        for j in range(i, len(X_dev_example[0])):
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_dev_example[:, i], X_dev_example[:, j],
                       labels - predictions)
            ax.set_xlabel(str(i))
            ax.set_ylabel(str(j))
            ax.set_zlabel("residual")
            plt.savefig('{}/{}-and-{}.png'.format(directory, i, j))
            plt.savefig(directory + str(i) + "and" + str(j) + ".png")
            plt.clf()
    print "=========== done plotting"


def main():
    print(train.head())
    # plot_feature_histograms("../visualizations/histograms")
    # plot_correlations("../visualizations/correlation")


main()
