import os
import errno
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

from calculate_stats import CalculateStats
from linear_regression import LinearRegression
from multilayer_perceptron import MultilayerPerceptron
from random_forest import RandomForest

DATASET_PATH = "../datasets/train.csv"
MODELS_DIR = "../models"
RESOURCES_PATH = "../resources/"


def clean_data(path):
    data_frame = pd.read_csv(path)

    # fill missing data for numeric features
    numeric_features = data_frame.select_dtypes(include=[np.number])

    for feature in numeric_features:
        data_frame[feature].fillna(data_frame[feature].mean(), inplace=True)

    # convert to numeric
    non_numeric_features = data_frame.select_dtypes(exclude=[np.number])

    for feature in non_numeric_features:
        mapping = {value: i for i, value in enumerate(data_frame[feature].unique())}
        data_frame[feature] = data_frame[feature].replace(
            mapping.keys(), mapping.values()
        )

    # dissregard unimportant features
    corr = data_frame.select_dtypes(include=[np.number]).corr()
    for column in (corr["SalePrice"].sort_values(ascending=False)[6:]).to_frame().T:
        data_frame.drop([column], axis=1, inplace=True)

    save_file_name = os.path.dirname(path) + os.sep + "house_prices_cleaned.csv"
    data_frame.to_csv(save_file_name, encoding="utf-8", index=False)

    return save_file_name


def split_data(path):
    data_frame = pd.read_csv(path)

    x = data_frame.loc[:, data_frame.columns != "SalePrice"]
    y = data_frame.loc[:, data_frame.columns == "SalePrice"]

    train_test_data = train_test_split(x, y, test_size=1 / 3, random_state=85)

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

    return [model(path, features_path, labels_path).get_path() for model in models]


def compare_results(models_paths, save_path, features_path, labels_path):

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "Latency": [],
    }

    for path in models_paths:
        model = joblib.load(path)

        start = time()
        prediction = model.predict(features)
        end = time()

        results["Model"].append(os.path.splitext(os.path.basename(path))[0])
        results["Accuracy"].append(round(accuracy_score(labels, prediction.round()), 3))
        results["Precision"].append(
            round(
                precision_score(labels, prediction.round(), average="micro"),
                3,
            )
        )
        results["Recall"].append(
            round(
                recall_score(labels, prediction.round(), average="micro"),
                3,
            )
        )
        results["Latency"].append(round((end - start) * 1000, 1))

    df = pd.DataFrame(results)

    fig, ax = CalculateStats.render_mpl_table(df, header_columns=0)
    fig.savefig(os.path.join(save_path, "model_comparison.png"))


def main():

    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), DATASET_PATH)

    clean_data_path = clean_data(DATASET_PATH)

    CalculateStats(clean_data_path)
    train_features, test_feature, train_labels, test_labels = split_data(
        clean_data_path
    )

    models = [
        LinearRegression,
        MultilayerPerceptron,
        RandomForest,
    ]

    results_paths = train_models(models, MODELS_DIR, train_features, train_labels)
    compare_results(results_paths, RESOURCES_PATH, test_feature, test_labels)


if __name__ == "__main__":
    main()
