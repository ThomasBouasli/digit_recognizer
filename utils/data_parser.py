import pandas as pd
import numpy as np

class DataParser:
    @staticmethod
    def parse(file_name):
        data = pd.read_csv(file_name)
        data = np.array(data)
        m, n = data.shape
        data_train = data.T
        y_train = data_train[0]
        x_train = data_train[1:n] / 255
        return x_train, y_train
