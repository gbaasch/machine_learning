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
    type = Preprocess[type]
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


def train_predict_evaluate(X_train, X_test, y_train, y_test, type_arr):
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


def run(train_X, train_y, val_X, val_y):
    train_X, val_X, train_y, val_y = clean_data(
        train_X, val_X, train_y, val_y, HandleMissing["D"]
    )

    train_X, val_X = preprocess_data(
        train_X, val_X, "MM"
    )

    # train_X = generate_features(train_X,)
    # val_X = generate_features(val_X)

    train_predict_evaluate(
        train_X, val_X, train_y, val_y, ["LR", "DT", "ADA"]
    )

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
