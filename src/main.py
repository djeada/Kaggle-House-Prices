import os
import errno
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
)
from time import time

from src.preprocessing.explore_dataset import CalculateStats
from src.models.linear_regression import LinearRegression
from src.models.multilayer_perceptron import MultilayerPerceptron
from src.models.random_forest import RandomForest

DATASET_PATH = "../datasets/train.csv"
MODELS_DIR = "../models"
RESOURCES_PATH = "../resources/"


def train_models(models, path, features_path, labels_path):

    return [model(path, features_path, labels_path).get_path() for model in models]


def main():

    ### Preprocessing (input dataset -> preprcoessing -> ready dataset)

    ### Models

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
