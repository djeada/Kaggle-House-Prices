import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "../datasets/train.csv"
MODELS_DIR = "../models"
RESOURCES_PATH = "../resources/"

from calculate_stats import CalculateStats


def clean_data(path):
    data_frame = pd.read_csv(path)

    # fill missing data for numeric features
    numeric_features = data_frame.select_dtypes(include=[np.number])

    for feature in numeric_features:
        data_frame[feature].fillna(data_frame[feature].mean(), inplace=True)

    # convert to numeric
    non_numeric_features = data_frame.select_dtypes(exclude=[np.number])
    
    for feature in non_numeric_features:
        mapping = {value : i for i, value in enumerate(data_frame[feature].unique())}
        data_frame[feature] = data_frame[feature].replace(mapping.keys(), mapping.values())
  

    # dissregard unimportant features
    data_frame.drop(["Id"], axis=1, inplace=True)

    save_file_name = os.path.dirname(path) + os.sep + "house_prices_cleaned.csv"
    data_frame.to_csv(save_file_name, encoding="utf-8", index=False)

    return save_file_name


def split_data(path):
    data_frame = pd.read_csv(path)

    x = data_frame.loc[:, data_frame.columns != "SalePrice"]
    y = data_frame.loc[:, data_frame.columns == "SalePrice"]

    train_test_data = train_test_split(x, y, test_size=1/3, random_state=85)

    dir_path = os.path.dirname(path) + os.sep

    paths = [
        dir_path + file_name
        for file_name in [
            "train_features.csv",
            "test_features.csv",
            "train_labels.csv",
            "test_labels.csv",
        ]
    ]

    for data, path in zip(train_test_data, paths):
        data.to_csv(path, index=False)

    return paths


def train_models(models, path, features_path, labels_path):
    pass


def compare_results(models_paths, save_path, features_path, labels_path):
    pass


def main():

    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), DATASET_PATH)

    clean_data_path = clean_data(DATASET_PATH)

    CalculateStats(clean_data_path)
    train_features, test_feature, train_labels, test_labels = split_data(
        clean_data_path
    )

if __name__ == "__main__":
    main()
