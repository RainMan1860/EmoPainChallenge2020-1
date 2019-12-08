from sklearn.decomposition import PCA


# compute the PCA and return the reduced data
def pca(X):
    pca_value = PCA()
    pca_value.fit(X)
    return pca_value.transform(X)
