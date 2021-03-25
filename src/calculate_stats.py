import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

RESOURCES_PATH = "../resources/"

class CalculateStats:
    def __init__(self, path):

        if not os.path.exists(RESOURCES_PATH):
            os.makedirs(RESOURCES_PATH)

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
        plt.title("Number of Houses vs House Prices")
        plt.xlabel("House Prices")
        plt.ylabel("Number of Houses")
        plt.savefig(os.path.join(RESOURCES_PATH, "number_of_houses_vs_house_prices.png"))
        plt.close("all")

    @staticmethod
    def numeric_features_correlation(df):
        numeric_features = df.select_dtypes(include=[np.number])
        corr = numeric_features.corr()

        df = (corr["SalePrice"].sort_values(ascending=False)[1:6]).to_frame().T
        df = df.round(2)
        fig, ax = CalculateStats.render_mpl_table(df)
        fig.savefig(os.path.join(RESOURCES_PATH, "numeric_features_correlation.png"))
        plt.close("all")

        return list(df.columns.values)

    @staticmethod
    def plot_sale_price_vs_highly_correlated_features(df, feature_list):

        for feature in feature_list:
            x, y = df[feature].values, df["SalePrice"].values
            m, b = np.polyfit(x, y, 1)

            plt.plot(x, m*x + b, color="#e41a1c", label="y={}x + {}".format(int(m), int(b)))

            plt.scatter(x=x, y=y)
            plt.title("Sale Price vs {}".format(feature))
            plt.ylabel("Sale Price")
            plt.xlabel(feature)
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(RESOURCES_PATH, "sale_price_vs_{}.png".format(feature)), bbox_inches='tight')
            plt.close("all")
