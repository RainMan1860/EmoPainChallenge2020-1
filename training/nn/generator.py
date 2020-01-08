import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random


def generatorCSVBatch(X_path, y_path, add_const="", drop_const="", header=None, ohe=False, batch_size=None):
    while True:
        data_files = os.listdir(y_path)
        while len(data_files) > 0:
            current_file = data_files.pop(random.randrange(len(data_files)))
            X = pd.read_csv(X_path + current_file.replace(drop_const, add_const), header=header)
            y = pd.read_csv(y_path + current_file, header=header)
            if ohe is not False:
                enc = OneHotEncoder(n_values=ohe)
                y = enc.fit_transform(y).toarray()
            # print("Data shape: {}x{}".format(X.shape[0], X.shape[1]))
            # print("Labels shape: {}x{}".format(y.shape[0], y.shape[1]))
            current_index = 0
            if batch_size is not None:
                while current_index < X.shape[0]:
                    sub_X = X[current_index:current_index+batch_size]
                    sub_y = y[current_index:current_index+batch_size]
                    # print("Sub data shape: {}x{}".format(sub_X.shape[0], sub_X.shape[1]))
                    # print("Sub labels shape: {}x{}".format(sub_y.shape[0], sub_y.shape[1]))
                    current_index = current_index + batch_size
                    yield sub_X, sub_y
            else:
                yield X, y
