import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

RESOURCES_PATH = "../resources/"


class CalculateStats:
    def __init__(self, path):
        df = pd.read_csv(path)
        CalculateStats.plot_number_of_houses_vs_house_prices(df)
        feature_list = CalculateStats.numeric_features_correlation(df)
        CalculateStats.plot_sale_price_vs_highly_correlated_features(df, feature_list)

    @staticmethod
    def render_mpl_table(
        data,
        col_width=3.0,
        row_height=0.625,
        font_size=14,
        header_color="#40466e",
        row_colors=["#f1f1f2", "w"],
        edge_color="w",
        bbox=[0, 0, 1, 1],
        header_columns=0,
        ax=None,
        **kwargs
    ):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array(
                [col_width, row_height]
            )
            fig, ax = plt.subplots(figsize=size)
            ax.axis("off")
            mpl_table = ax.table(
                cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs
            )
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(font_size)

            for k, cell in mpl_table._cells.items():
                cell.set_edgecolor(edge_color)

                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight="bold", color="w")
                    cell.set_facecolor(header_color)
                else:
                    cell.set_text_props(ha="center")
                    cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        return ax.get_figure(), ax

    @staticmethod
    def plot_number_of_houses_vs_house_prices(df):
        desc = df.SalePrice.describe().T
        desc = desc.round(2)
        fig = plt.hist(df.SalePrice)
        plt.title("Mean")
        plt.xlabel("value")
        plt.ylabel("Frequency")
        plt.savefig("abc.png")
        plt.close("all")

    @staticmethod
    def numeric_features_correlation(df):
        numeric_features = df.select_dtypes(include=[np.number])
        corr = numeric_features.corr()

        df = (corr["SalePrice"].sort_values(ascending=False)[1:6]).to_frame().T
        df = df.round(2)
        fig, ax = CalculateStats.render_mpl_table(df)
        fig.savefig("table.png")
        plt.close("all")

        return list(df.columns.values)


    @staticmethod
    def plot_sale_price_vs_highly_correlated_features(df, feature_list):

        for feature in feature_list:
            x, y = df[feature].values, df["SalePrice"].values
            m, b = np.polyfit(x, y, 1)

            plt.plot(x, m*x + b, color="#e41a1c")


            plt.scatter(x=x, y=y)
            plt.ylabel("Sale Price")
            plt.xlabel(feature)
            plt.savefig("sale_price_vs_{}.png".format(feature))
            plt.close("all")
