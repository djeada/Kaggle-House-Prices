import pandas as pd

from src.models.linear_regression import LinearRegression
from src.models.multilayer_perceptron import MultilayerPerceptron
from src.models.random_forest import RandomForest
from src.postprocessing.prediction_table import create_prediction_table
from src.preprocessing.clean_dataset import clean_data
from src.preprocessing.split_dataset import split_dataset

DATASET_PATH = "../data/train.csv"
LABELS_HEADERS = ["SalePrice"]


def main():

    ### Preprocessing (input dataset -> preprcoessing -> ready dataset)
    print("Preprocessing...")

    raw_dataset = pd.read_csv(DATASET_PATH)

    print("Cleaning dataset...")
    clean_dataset = clean_data(raw_dataset)
    print("Dataset cleaned.")

    # Split the dataset into train and test sets

    print("Splitting dataset...")
    x_dataset = clean_dataset.drop(LABELS_HEADERS, axis=1)
    y_dataset = clean_dataset[LABELS_HEADERS]

    dataset_split = split_dataset(x_dataset, y_dataset, save_to_file=True)

    print("Preprocessing finished.")

    ### Training (ready dataset -> train -> model)
    print("Training...")

    model_types = [
        LinearRegression,
        MultilayerPerceptron,
        RandomForest,
    ]

    models = []

    for model_type in model_types:
        print(f"Training {model_type.__name__}...")
        model = model_type()
        model.fit(dataset_split.train_x, dataset_split.train_y)
        model.save(f"../output/{model.__class__.__name__}.joblib")
        models.append(model)
    print("Training finished.")

    ### Postprocessing (model -> postprocessing -> metrics)
    print("Postprocessing...")
    for model in models:
        predicted_y_array = model.predict(dataset_split.test_x)
        predicted_y = pd.DataFrame(
            predicted_y_array,
            columns=[f"{header}_predicted" for header in LABELS_HEADERS],
        )
        create_prediction_table(
            dataset_split.test_x, dataset_split.test_y, predicted_y, save_to_file=True
        )

    print("Postprocessing finished.")


if __name__ == "__main__":
    main()
