from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import csv
from sklearn import tree
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
import numpy as np
import math


def DummyCode(x, feature, numCategories, most): #takes an observation, and a features and the numCategories
    newFeatures = np.zeros(numCategories)
    if feature==7 or feature ==9:
        x[feature] -= 1
    for category in range(0,numCategories):
        if x[feature]==most:
            pass#default value
        elif x[feature] == category:
            newFeatures[category]=1
    x = np.append(x, newFeatures,axis=0)
    return x


def clean_training_data(X, y):
  """
  This function can be used to modify the training data in any way
  (e.g. removing outliers).
  Please note that you should not modify the number of columns in X,
  although you could choose to drop rows in X and y.
  :param X: an n x p design matrix
  :param y: a length-n response vector
  :return X: a modified design matrix (with p columns)
  : return y: a modified response vector
  """
  return X, y


# function for plotting individual features against y
def NormalizeAndPlot(X,Y,og):
    for i in range(0, len(X[0])-1):
        plt.scatter(X[:,i],Y)
        plt.title(og.columns[i])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('X'+str(i)+".png")
        print("plot X"+str(i)+".png made")
        plt.clf()
        plt.hist(X[:,i])
        plt.xlabel("X"+str(i))
        plt.savefig("X_hist"+str(i)+".png")
        plt.clf()
    plt.hist(Y)
    plt.xlabel("Y")
    plt.savefig("Y_hist.png")


def transform(x):
    """
    Apply a transformation to a feature vector for a single instance
    :param x: a feature vector for a single instance
    :return: a modified feature vector
    """
    # 8 is logarithmically distributed. This makes it categorical,
    # the one is added to avoid taking the log of 0. dividing by 1 makes it an even int for the categories

    return x


def train(X, y):
    """ Train a model

    :param X: n x p design matrix
    :param y: response vector of length n
    :return weights: weight vector of length p
    """
    n, p = X.shape
    print p
    weights = np.zeros(p)

    reg = linear_model.LogisticRegression(normalize=True)
    reg.fit(X,y)
    constant = reg.intercept_
    wHat = reg.coef_
    print np.shape(wHat)
    wHat[len(wHat)-1] = constant # set constant feature to the intercept of the model
    return wHat


def predict(X, weights):
    """
    This function will be called to make predictions.
    Please leave it as it is.
    :param X: n x p design matrix
    :param weights: weight vector of length p
    :return y: lenght-n vector of predictions
    """
    y = np.dot(X, weights)
    return y


def MSE(y, predictions):
    """
    This function computes the MSE.
    Please leave it as it is.
    :param y: a vector of true responses
    :param predictions: a vector of predicted responses
    :return mse: The mean-squared error of the predictions
    """

    mse = np.mean((y - predictions)**2)
    return mse


def train_and_predict(X_train, y_train, X_new):
    """
    This function will train a model and make predictions on new data.
    This is the function that will be called by the unit tests
    Please leave it as it is.
    :param X_train: n_train x p design matrix of training data
    :param y_train: length n_train vector of training responses
    :param X_new: n_new x p design matrix to test on
    :return predictions_train: vector of predictions on training data
    :return predictions_new: vector of predictions on new data
    """
    # clean the training data
    X_train_clean, y_train_clean = clean_training_data(X_train, y_train)

    # transform the training data
    X_train_transformed = np.vstack([transform(x) for x in X_train_clean])
    #NormalizeAndPlot(X_train_transformed,y_train_clean)
    # transform the new data
    X_new_transformed = np.vstack([transform(x) for x in X_new])
    #NormalizeAndPlot(X_train_transformed,y_train_clean)
    # learn a model
    weights = train(X_train_transformed, y_train_clean)
    # make predictions on the training data
    predictions_train = predict(X_train_transformed,weights)
    # make predictions on the new data

    predictions = predict(X_new_transformed, weights)
    # report the MSE on the training data
    train_MSE = MSE(y_train_clean, predictions_train)
    print("MSE on training data = %0.4f" % train_MSE)
    # return the predictions on the new data
    return predictions


def plot_data():
    diabetes = pd.read_csv("diabetes.csv", index_col=0)
    print(diabetes.head())
    # print(diabetes.describe(include='all'))
    # print(diabetes.columns)
    print(diabetes.dtypes)
    # print(list(diabetes))
    print("Done")


def run_train_and_predict():
    # load the data:  unfortunately it has to be provided here because
    # the multi-file interface prevents the console from working properly
    f = open("train.csv", 'r')
    line = f.readline().strip()
    header = line
    X_train = []
    line = f.readline().strip()
    while line != "":
        row = line.split(",")
        row = [float(x) for x in row]
        X_train.append(row)
        line = f.readline().strip()

    X_train = np.array(X_train)
    y_train = X_train[:,-1]
    n, p = X_train.shape


    #X_train = np.random.rand(n,p)
    #n = np.shape(y_train)
    #print n
    #y_train = np.random.rand(n[0],)


    # create a fake data matrix to test the functions
    X_dev_example = X_train[300:-2]
    y_dev_example = y_train[300:-2]
    X_train = X_train[:300]
    y_train = y_train[:300]

    # call train_and_predict to make sure things are working
    # Uncomment this to test:
    predictions = train_and_predict(X_train, y_train, X_dev_example)
    test_MSE = MSE(y_dev_example, predictions)
    print("MSE on test data = %0.4f" % test_MSE)

    X_dev_example = np.delete(X_dev_example,37,0)
    y_dev_example = np.delete(y_dev_example,37,0)
    predictions = np.delete(predictions,37,0)

"""
print(np.argmin(y_dev_example-predictions))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_dev_example[:,7],X_dev_example[:,4],y_dev_example-predictions)
ax.set_xlabel("7")
ax.set_ylabel(str(4))
ax.set_zlabel("residual")
for angle in range(0,1080):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)

plt.savefig("ExponentialRotation"+".png")
plt.clf()
"""
"""X_dev_example = np.vstack([transform(x) for x in X_dev_example])


def plotFeaturesAgainstResidual(X_dev_example,y_dev_example,predictions):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    for i in range(0,len(X_dev_example[0])):
        for j in range(i,len(X_dev_example[0])):
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_dev_example[:,i],X_dev_example[:,j],y_dev_example-predictions)
            ax.set_xlabel(str(i))
            ax.set_ylabel(str(j))
            ax.set_zlabel("residual")
            plt.savefig(str(i)+"and"+str(j)+".png")
            plt.clf()



index = np.argmin(y_dev_example-predictions)

predictions = np.delete(predictions,index)
y_dev_example = np.delete(y_dev_example, index)
X_dev_example = np.delete(X_dev_example,index, axis=0)
"""
#X_dev_example = np.vstack([transform(x) for x in X_dev_example])

#plotFeaturesAgainstResidual(X_dev_example,y_dev_example,predictions)
"""
plt.scatter(predictions, y_dev_example-predictions,  color='black')
plt.xlabel("Predictions")
plt.ylabel("Residual")
plt.hlines(y=0,xmin=-2,xmax=2)

plt.xticks(())
plt.yticks(())
plt.savefig("residual.png")"""


def parse_x_y(data):
    # do two sets of square braces allow you to extract as data frame?
    #  would be useful for plotting
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x.iloc[:, 1:] = x.iloc[:, 1:].replace(0, np.nan)
    return x, y


def clean_data(X_train, X_val, y_train, y_val, type):
    if type == 'DROP':
        X_train, y_train = drop_rows_with_missing(X_train, y_train)
        X_val, y_val = drop_rows_with_missing(X_val, y_val)
    elif type == 'IMPUTE':
        X_train = impute_values(X_train)
        X_val = impute_values(X_val)
    elif type == 'IMPUTE_EXTEND':
        X_train = impute_values_extended(X_train)
        X_val = impute_values_extended(X_val)

    return X_train, X_val, y_train, y_val


def preprocess_data(X_train, X_val, type):
    if type == "SCALE":
        X_train = preprocessing.scale(X_train)
        X_val = preprocessing.scale(X_val)
    elif type == "MAXABS":
        X_train = preprocessing.maxabs_scale(X_train)
        X_val = preprocessing.maxabs_scale(X_val)
    elif type == "MINMAX":
        X_train = preprocessing.minmax_scale(X_train)
        X_val = preprocessing.minmax_scale(X_val)

    return X_train, X_val


def impute_values(data):
    my_imputer = Imputer()
    return my_imputer.fit_transform(data)


def impute_values_extended(data):
    # make copy to avoid changing original data (when Imputing)
    new_data = data.copy()

    # make new columns indicating what will be imputed
    cols_with_missing = [col for col in new_data.columns
                         if new_data[col].isnull().any()]
    for col in cols_with_missing:
        new_data[col + '_was_missing'] = new_data[col].isnull()

    # Imputation
    my_imputer = Imputer()
    return my_imputer.fit_transform(new_data)


def drop_rows_with_missing(x, y):
    missing_indexes = []
    for index, row in x.iterrows():
        for i in range(0, len(row)):
            if pd.isna(row.iloc[i]):
                missing_indexes.append(index)

    x = x.drop(missing_indexes)
    y = y.drop(missing_indexes)
    return x, y


def visualize(dataframe):
    plt.matshow(dataframe.corr()) # correliation coefficient
    # TODO plot histograms
    # TODO correlation grouped by outcome, maybe as heatmap?
    return


def compare_models():
    # TODO plot confidence intervals, accuracy, confusion matrix
    return


def select_features(X):
    x = np.append(X[:,1].reshape(-1, 1), X[:,6].reshape(-1, 1), axis=1)
    x = np.append(x, X[:, 7].reshape(-1, 1), axis=1)
    x = np.append(x, X[:, 15].reshape(-1, 1), axis=1)
    x = np.append(x, X[:, 31].reshape(-1, 1), axis=1)
    x = np.append(x, X[:, 29].reshape(-1, 1), axis=1)
    return x


def generate_features(X):
    X_new = X.copy()
    for i in range(X.shape[1] - 1):
        j = i
        while j < X.shape[1] - 1:
            a = X[:,i] * X[:,j]
            a = a.reshape(-1, 1)
            X_new = np.append(X_new, a, axis=1)
            j += 1
    return X_new


def run_log_ref(X_train, X_test, y_train, y_test, type_arr):
    for x in type_arr:
        type = Model[x]
        if type == "LOGISTIC_REGRESSION":
            model = linear_model.LogisticRegression()
        if type == "SVC":
            model = sklearn.svm.SVC()
        if type == "DECISION_TREE":
            model = tree.DecisionTreeClassifier(max_depth=2)
        if type == "ADABOOST":
            model = ensemble.AdaBoostClassifier(n_estimators=3)
        if type == "GRADIENT_BOOSTING_REGRESSOR":
            model = ensemble.GradientBoostingRegressor()

        fit = model.fit(X_train, y_train)

        if type == "DECISION_TREE":
            tree.export_graphviz(fit, out_file='tree.dot')

        predictions_train = model.predict(X_train)
        predictions_val = model.predict(X_test)

        tn, fp, fn, tp = metrics.confusion_matrix(
            predictions_val, y_test
        ).ravel()

        with open(type + "_predictions.csv", "w") as writefile:
            wf = csv.writer(writefile)
            wf.writerow(predictions_train)

        # print("True Negative: {} \n False Postitve: {} \n False Negative: {} \n "
        #       "True Positive: {}".format(tn, fp, fn, tp))
        n = fp + fn
        d = fp + fn + tp + tn
        ce = (n / d) * 100
        print("===================== TYPE: {} ==================".format(type))
        print("Classification Error: {} %".format(ce))

        print("MSE on training data ", MSE(y_train, predictions_train))
        print("MSE on test data ", MSE(y_test, predictions_val))
        print("\n")
        print("error on training data", 1 - model.score(X_train, y_train))
        print("error on test data", 1 - model.score(X_test, y_test))
        print("\n\n")


def run(training_csv, validation_csv):
    # run_train_and_predict():
    # plot_data()
    train = pd.read_csv(training_csv)
    validate = pd.read_csv(validation_csv)

    print(train.info())

    X_train, y_train = parse_x_y(train)
    X_val, y_val = parse_x_y(validate)

    visualize(train)

    X_train, X_val, y_train, y_val = clean_data(
        X_train, X_val, y_train, y_val, HandleMissing["D"]
    )

    X_train, X_val = preprocess_data(
        X_train, X_val, Preprocess["MM"]
    )

    X_train = generate_features(X_train,)
    X_val = generate_features(X_val)
    #
    # print("SHAPE: " + str(X_train.shape[0]) + " " + str(X_train.shape[1]))
    X_train = select_features(X_train)
    X_val = select_features(X_val)

    run_log_ref(
        X_train, X_val, y_train, y_val, ["LR", "DT", "ADA"]
    )

    # NormalizeAndPlot(X_train, y_train, train)


HandleMissing = {
    "D": "DROP",
    "I": "IMPUTE",
    "IE": "IMPUTE_EXTEND"
}

Preprocess = {
    "S": "SCALE",
    "MA": "MAXABS",
    "MM": "MINMAX"
}

Model = {
    "LR": "LOGISTIC_REGRESSION",
    "DT": "DECISION_TREE",
    "ADA": "ADABOOST",
    "GB": "GRADIENT_BOOSTING_REGRESSOR"
}
