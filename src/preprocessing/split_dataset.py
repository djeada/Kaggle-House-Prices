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
