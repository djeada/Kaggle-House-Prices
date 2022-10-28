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
    data_frame.drop(["Id"], axis=1, inplace=True)

    save_file_name = os.path.dirname(path) + os.sep + "house_prices_cleaned.csv"
    data_frame.to_csv(save_file_name, encoding="utf-8", index=False)

    return save_file_name
