from abc import abstractmethod, ABC
from collections import Counter
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


class BaseFilter(ABC):
    @abstractmethod
    def run(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Run the filter on the given data frame.

        :param data_frame: The data frame to run the filter on.
        :return: The filtered data frame.
        """
        pass


class FillMissingValuesFilter(BaseFilter):
    def run(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the mean of the column.

        :param data_frame: The data frame to run the filter on.
        :return: The filtered data frame.
        """

        for column in data_frame.columns:
            data_frame[column].fillna(data_frame[column].mean(), inplace=True)

        return data_frame


class DropFeaturesFilter(BaseFilter):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def run(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the given columns from the data frame.

        :param data_frame: The data frame to run the filter on.
        :return: The filtered data frame.
        """
        return data_frame.drop(self.columns_to_drop, axis=1)


class EncodeCategoricalVariablesFilter(BaseFilter):
    def run(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Converts non-numeric features to numeric.

        :param data_frame: The data frame to run the filter on.
        :return: The filtered data frame.
        """
        non_numeric_features = data_frame.select_dtypes(exclude=[np.number])

        for feature in non_numeric_features:
            mapping = {value: i for i, value in enumerate(data_frame[feature].unique())}
            data_frame[feature] = data_frame[feature].replace(
                mapping.keys(), mapping.values()
            )

        return data_frame


class RemoveOutliersFilter(BaseFilter):
    def detect_outliers(self, data_frame: pd.DataFrame, n=5) -> List[int]:
        numerical_features = data_frame.select_dtypes(include=[np.number]).columns
        outlier_indices = []
        for feature in numerical_features:
            q1 = np.percentile(data_frame[feature], 25)
            q3 = np.percentile(data_frame[feature], 75)
            iqr = q3 - q1
            outlier_step = 1.5 * iqr
            outlier_list_col = data_frame[
                (data_frame[feature] < q1 - outlier_step)
                | (data_frame[feature] > q3 + outlier_step)
            ].index
            outlier_indices.extend(outlier_list_col)

        result = [i for i in set(outlier_indices) if outlier_indices.count(i) > n]
        return result

    def run(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the given columns from the data frame.

        :param data_frame: The data frame to run the filter on.
        :return: The filtered data frame.
        """
        outliers_to_drop = self.detect_outliers(data_frame)
        return data_frame.drop(outliers_to_drop, axis=0)


def clean_data(
    data_frame: pd.DataFrame,
    filters_types: Iterable[BaseFilter] = (
        RemoveOutliersFilter,
        EncodeCategoricalVariablesFilter,
        FillMissingValuesFilter,
    ),
) -> pd.DataFrame:
    """
    Clean the given data frame using the given filters.

    :param data_frame: The data frame to clean.
    :param filters: The filters to use.
    :return: The cleaned data frame.
    """
    for clean_filter_type in filters_types:
        clean_filter = clean_filter_type()
        data_frame = clean_filter.run(data_frame)

    return data_frame
