from ml_utils.run_models import (preprocess_data, train_predict_evaluate)
from ml_utils.parse_data import parse_data


def main():
    with open("../train.csv") as f:
        (train_X, train_y, train_header) = parse_data(f, 1, "M")

    with open("../val.csv") as f:
        (val_X, val_y, val_header) = parse_data(f, 1, "M")

    train_predict_evaluate(
        train_X, val_X, train_y, val_y,
        ["LR", "SVC", "DT", "ADA"])

    print("DONE")


main()
