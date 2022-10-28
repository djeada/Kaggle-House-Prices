def calculate_r2_score(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the R^2 score.

    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The R^2 score.
    """
    return r2_score(test_y, predicted_y)


def calculate_rmse(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the RMSE.

    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The RMSE.
    """
    return np.sqrt(mean_squared_error(test_y, predicted_y))


def calculate_nrmse(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the NRMSE.

    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The NRMSE.
    """
    return np.sqrt(mean_squared_error(test_y, predicted_y)) / np.std(predicted_y)


def calculate_mae(test_y: np.ndarray, predicted_y: np.ndarray) -> float:
    """
    Calculates the MAE.

    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :return: The MAE.
    """
    return mean_absolute_error(test_y, predicted_y)


def compare_models(
    models: Iterable[BaseModel],
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    save_to_file: bool = False,
) -> pd.DataFrame:
    """
    Compares the performance of the given models.

    :param models: The models to compare.
    :param test_x: The test input data.
    :param test_y: The test output data.
    :param save_to_file: Whether to save the table to file.
    :return: The comparison table.
    """
    scores = []
    for model in models:
        predicted_y = model.predict(test_x)
        scores.append(
            {
                "model": model.name,
                "r2_score": calculate_r2_score(test_y, predicted_y),
                "rmse": calculate_rmse(test_y, predicted_y),
                "nrmse": calculate_nrmse(test_y, predicted_y),
                "mae": calculate_mae(test_y, predicted_y),
            }
        )

    comparison_table = pd.DataFrame(scores)
    if save_to_file:
        comparison_table.to_csv("data/comparison_table.csv")

    return comparison_table
