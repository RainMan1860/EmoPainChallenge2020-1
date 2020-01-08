from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import os
import pandas as pd
import numpy as np
from training.preprocess import *
from sklearn.utils.class_weight import compute_class_weight


# multi-class SVM classifier that implements the "one-against-one" approach
# decision_function_shape:
#   "ovr": one-vs-rest
#   "ovo": one-vs-one
# kernel: "rbf", "linear", "poly", "sigmoid", "precomputed"
# class_weight: dictionary of the form {class_label : value}, where value is a fp number > 0 that sets the parameter
#               c of class class_label to c * value.
#               Sets the parameter to "balanced" to sets the weights to n_samples / (n_classes * np.bincount(y))
def multiclassSVM(X, y, decision_function_shape="ovr", kernel="rbf", class_weight=None):
    clf = svm.SVC(decision_function_shape=decision_function_shape, kernel=kernel, class_weight=class_weight)
    clf.fit(X, y)
    return clf


# multi-class SVM classifier that implements the "one-vs-the-rest" approach
# class_weight: dictionary of the form {class_label : value}, where value is a fp number > 0 that sets the parameter
#               c of class class_label to c * value.
#               Sets the parameter to "balanced" to sets the weights to n_samples / (n_classes * np.bincount(y))
def multiclassLinearSVM(X, y, class_weight=None):
    clf = svm.LinearSVC(class_weight=class_weight)
    clf.fit(X, y)
    return clf


# random forest classifier
#   num_trees: number of trees in the forest
# class_weight: dictionary of the form {class_label : value}, where value is a fp number > 0 that sets the parameter
#               c of class class_label to c * value.
#               Sets the parameter to "balanced" to sets the weights to n_samples / (n_classes * np.bincount(y))
#               Sets the parameter to "balanced_subsample" to compute the weights on the base of the bootstrap sample
#               for every tree grown
def randomForest(X, y, num_trees=100, class_weight=None):
    clf = RandomForestClassifier(n_estimators=num_trees, class_weight=class_weight)
    clf.fit(X, y)
    return clf


# SGD Classifier with partial fit
#   loss: the loss function to be used
#   penalty: regularization term
def SGD_classifier(X_path, y_path, add_const_name="", drop_const_name="", loss="hinge", penalty="l2", alpha=0.0001,
                   max_iter=None, tol=None, shuffle=True, n_iter_no_change=5, class_weight=None, classes=["0", "1"]):
    if class_weight == "balanced":
        y_full = concat_csv(y_path)
        y_full[0] = y_full[0].astype(str)
        # print(y_full[0].dtypes)
        # class_weight = compute_class_weight("balanced", np.unique(y_full.values), y_full)
        class_weight = compute_class_weight("balanced", classes, y_full.values.ravel())
        class_weight_dict = {}
        for i in range(len(class_weight)):
            class_weight_dict[str(i)] = class_weight[i]
        print("Computed weights for classes:")
        print(class_weight_dict)

    sgd = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol, shuffle=shuffle,
                        n_iter_no_change=n_iter_no_change, class_weight=class_weight_dict)
    for df_label_path in os.listdir(y_path):
        y = pd.read_csv(y_path + df_label_path, header=None)
        y[0] = y[0].astype(str)
        df_path = df_label_path.replace(drop_const_name, add_const_name)
        X = pd.read_csv(X_path + df_path, header=None)
        print("Training on file {}".format(df_path))
        sgd.partial_fit(X, y.values.ravel(), classes=classes)

    return sgd
