from ml_utils.run_models import (preprocess_data, train_predict_evaluate)


def main():
    train_X = [[]]
    train_y = []
    val_X = [[]]
    val_y = []

    # TODO preprocess_data(train_X, val_X, ["MM"])

    train_predict_evaluate(
        train_X, train_y, val_X, val_y, ["LR", "DT", "ADA"])
    print("DONE")


main()
