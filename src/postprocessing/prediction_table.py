import pandas as pd


def create_prediction_table(
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    predicted_y: pd.DataFrame,
    save_to_file: bool = False,
) -> pd.DataFrame:
    """
    Creates a table with the actual and predicted values.

    :param test_x: The test input data.
    :param test_y: The test output data.
    :param predicted_y: The predicted output data.
    :param save_to_file: Whether to save the table to file.
    :return: The prediction table.
    """
    difference_between_test_and_predicted = test_y - predicted_y

    prediction_table = pd.concat(
        [test_x, test_y, predicted_y, difference_between_test_and_predicted], axis=1
    )

    if save_to_file:
        prediction_table.to_csv("../output/prediction_table.csv")

    return prediction_table
