from ml_utils.run_models import (preprocess_data, train_predict_evaluate)
from ml_utils.parse_data import parse_data
from ml_utils.visualize import plot_features_against_residuals

VIS_RESIDUALS_DIR = "../visualizations/residuals"


def main():
    with open("../train.csv") as f:
        (train_X, train_y, train_header) = parse_data(f, 1, "M")

    with open("../val.csv") as f:
        (val_X, val_y, val_header) = parse_data(f, 1, "M")

    predictions = train_predict_evaluate(
        train_X, val_X, train_y, val_y,
        ["LR", "SVC", "DT", "ADA"])

    plot_features_against_residuals(
        train_X, train_y, predictions, VIS_RESIDUALS_DIR)

    print("DONE")

main()
