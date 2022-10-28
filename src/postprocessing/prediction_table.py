import pandas as pd


def create_prediction_table(
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    predicted_y: pd.DataFrame,
    model_name: str,
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

    prediction_table = test_x.join([test_y, predicted_y])

    for expected_y_header, predicted_y_header in zip(
        test_y.columns, predicted_y.columns
    ):
        prediction_table[f"{expected_y_header}_diff"] = (
            prediction_table[expected_y_header] - prediction_table[predicted_y_header]
        )

    if save_to_file:
        prediction_table.to_csv(f"../output/prediction_table_{model_name}.csv")

    return prediction_table
