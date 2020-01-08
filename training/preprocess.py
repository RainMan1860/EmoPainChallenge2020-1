import os
from sklearn.decomposition import PCA
import pandas as pd


# compute the PCA and return the reduced data
def pca(X):
    pca_value = PCA()
    pca_value.fit(X)
    return pca_value.transform(X)


# join data from different csv
def concat_csv(X_directory, y_directory=None):
    X_list = []
    if y_directory is not None:
        y_list = []
    for X_file in os.listdir(X_directory):
        if X_file.endswith(".csv"):
            df_X = pd.read_csv(X_directory + X_file, header=None)
            X_list.append(df_X)
            if y_directory is not None:
                df_y = pd.read_csv(y_directory + X_file, header=None)
                y_list.append(df_y)

    X = pd.concat(X_list)
    if y_directory is not None:
        y = pd.concat(y_list)
        return X, y
    else:
        return X
