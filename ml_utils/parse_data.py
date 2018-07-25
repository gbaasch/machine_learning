import csv

import numpy as np


def parse_data(open_file, labels_col, label_val):
    """Reads data from an open data file and parses out the binary labels and features.

    Args:
        open_file (csv.reader): An open file containing the data
        labels_col (int): The column index containing the labels
        label_val (flexible): An example value of one of the labels

    Returns:
        data (Matrix[float]): The feature data
        labels (list[int]): The labels
        header (list[str]): The header values of the data

    """
    reader = csv.reader(open_file)
    header = next(reader)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)


    labels = list(data[:, labels_col])
    labels = [0 if x == label_val else 1 for x in labels]

    data = np.insert(data[:, labels_col + 1:], 0, data[:, 0:labels_col].transpose(), axis=1)

    data = data.astype(np.float)
    header = [header[0:labels_col]] + header[labels_col + 1:]

    return data, labels, header



