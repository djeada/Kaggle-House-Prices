import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

RESOURCES_PATH = "../resources/"


class CalculateStats:
    def __init__(self, path):
        df = pd.read_csv(path)
        desc = df.SalePrice.describe().T
        desc = desc.round(2)
        print(desc)

        plt.hist(df.SalePrice)
        plt.show()
