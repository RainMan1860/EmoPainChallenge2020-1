import pandas as pd
from classify import multiclassSVM, multiclassLinearSVM

path_hog_train = ""
path_hog_train_label = ""
path_hog_validation = ""
path_hog_validation_label = ""

hog_train = pd.read_csv(path_hog_train)
hog_train_label = pd.read_csv(path_hog_train_label)

hog_validation = pd.read_csv(path_hog_validation)
path_hog_validation_label = pd.read_csv(path_hog_validation_label)

# try without class_weight too
svm_ovr = multiclassSVM(hog_train, hog_train_label, decision_function_shape="ovr", class_weight="balanced")
svm_ovo = multiclassSVM(hog_train, hog_train_label, decision_function_shape="ovo", class_weight="balanced")
linear_svm = multiclassLinearSVM(hog_train, hog_train_label, class_weight="balanced")
