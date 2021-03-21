import os

DATASET_PATH = "../datasets/train.csv"
MODELS_DIR = "../models"
RESOURCES_PATH = "../resources/"

from calculate_stats import CalculateStats


def clean_data(path):
    pass


def split_data(path):
    pass


def train_models(models, path, features_path, labels_path):
    pass


def compare_results(models_paths, save_path, features_path, labels_path):
    pass


def main():

    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), DATASET_PATH)

    # clean_data_path = clean_data(DATASET_PATH)

    CalculateStats(DATASET_PATH)


if __name__ == "__main__":
    main()
