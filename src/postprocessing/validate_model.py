def compare_results(models_paths, save_path, features_path, labels_path):

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    results = {
        "Model": [],
        "R^2": [],
        "NRMSE": [],
        "Latency": [],
    }

    for path in models_paths:
        model = joblib.load(path)

        start = time()
        prediction = model.predict(features)
        end = time()

        results["Model"].append(os.path.splitext(os.path.basename(path))[0])
        results["R^2"].append(round(r2_score(labels, prediction.round()), 3))
        results["NRMSE"].append(
            round(
                np.sqrt(mean_squared_error(labels, prediction.round()))
                / np.std(prediction.round()),
                3,
            )
        )
        results["Latency"].append(round((end - start) * 1000, 1))

    df = pd.DataFrame(results)

    fig, ax = CalculateStats.render_mpl_table(df, header_columns=0)
    fig.savefig(os.path.join(save_path, "model_comparison.png"))
