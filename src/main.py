import numpy as np
import pandas as pd

from src.models.gradient_boost import GradientBoost
from src.models.lasso import Lasso
from src.models.linear_regression import LinearRegression
from src.models.multilayer_perceptron import MultilayerPerceptron
from src.models.random_forest import RandomForest
from src.postprocessing.prediction_table import create_prediction_table
from src.postprocessing.performance_metrics import (
    calculate_r2_score,
    calculate_rmse,
    calculate_nrmse,
)
from src.preprocessing.clean_dataset import (
    clean_data,
    EncodeCategoricalVariablesFilter,
    FillMissingValuesFilter,
)
from src.preprocessing.split_dataset import split_dataset

TRAIN_DATASET_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"
LABELS_HEADERS = ["SalePrice"]


def main():

    ### Preprocessing (input dataset -> preprcoessing -> ready dataset)
    print("Preprocessing...")
    output_dir = Path('../output')
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    raw_test_dataset = pd.read_csv(TEST_DATA_PATH)

    print("Cleaning dataset...")
    clean_dataset = clean_data(raw_dataset)
    clean_test_dataset = clean_data(
        raw_test_dataset,
        filters_types=(
            EncodeCategoricalVariablesFilter,
            FillMissingValuesFilter,
        ),
    )
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
        GradientBoost,
        Lasso,
    ]

    models = []

    for model_type in model_types:
        print(f"Training {model_type.__name__}...")
        model = model_type()
        model.fit(dataset_split.train_x.to_numpy(), dataset_split.train_y.to_numpy())
        model.save(f"../output/{model.__class__.__name__}.joblib")
        models.append(model)
    print("Training finished.")

    ### Postprocessing (model -> postprocessing -> metrics)
    print("Postprocessing...")
    scores = []
    for model in models:
        model_name = model.__class__.__name__
        test_y_array = dataset_split.test_y.to_numpy()
        predicted_y_array = model.predict(dataset_split.test_x)
        predicted_y = pd.DataFrame(
            predicted_y_array,
            columns=[f"{header}_predicted" for header in LABELS_HEADERS],
        )
        create_prediction_table(
            dataset_split.test_x,
            dataset_split.test_y,
            predicted_y,
            model_name,
            save_to_file=True,
        )
        score = {
            "model": model_name,
            "r2_score": calculate_r2_score(test_y_array, predicted_y_array),
            "rmse": calculate_rmse(test_y_array, predicted_y_array),
            "nrmse": calculate_nrmse(test_y_array, predicted_y_array),
        }
        scores.append(score)
        print(score)

    # Find the model with best r2_score
    best_model = max(scores, key=lambda score: score["r2_score"])
    print(f"Best model: {best_model['model']}")
    print("Postprocessing finished.")

    # use the best model on test data
    best_model = models[scores.index(best_model)]
    test_x_array = clean_test_dataset.to_numpy()
    predicted_y_array = np.atleast_2d(best_model.predict(test_x_array)).T

    output_data_frame = clean_test_dataset.filter(["Id"], axis=1)
    for i, header in enumerate(LABELS_HEADERS):
        output_data_frame[header] = predicted_y_array[:, i]

    output_data_frame.to_csv("../output/predictions.csv", index=False)


if __name__ == "__main__":
    main()
