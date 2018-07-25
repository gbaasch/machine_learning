import matplotlib.pyplot as plt

def plotFeaturesAgainstResidual(X_dev_example,y_dev_example,predictions):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    for i in range(0,len(X_dev_example[0])):
        for j in range(i, len(X_dev_example[0])):
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_dev_example[:, i], X_dev_example[:, j], y_dev_example-predictions)
            ax.set_xlabel(str(i))
            ax.set_ylabel(str(j))
            ax.set_zlabel("residual")
            plt.savefig(str(i)+"and"+str(j)+".png")
            plt.clf()