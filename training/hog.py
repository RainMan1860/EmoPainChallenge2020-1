from joblib import dump
from training.preprocess import *
from training.train import *

path_hog_train = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/hog/train/"
path_hog_train_label = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/labels/train/"

# path_hog_validation = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/HOG Features/valid/"
# path_hog_validation_label = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/valid/"

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

print("Training the SGD Classifier...")
sgd_class = SGD_classifier(path_hog_train, path_hog_train_label, max_iter=1000, class_weight="balanced", classes=classes)
dump(sgd_class, "C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/sgd/hog_sgd_classifier_balanced.joblib")
print("Done")

'''
print("Loading training set...")
X_train, y_train = concat_csv(path_hog_train, path_hog_train_label)
# X_val, y_val = concat_csv(path_hog_validation, path_hog_validation_label)

print("X_train {}x{}; y_train {}x{}".format(X_train.shape[0], X_train.shape[1], y_train.shape[0], y_train.shape[1]))
# print("X_val {}x{}; y_val {}x{}".format(X_val.shape[0], X_val.shape[1], y_val.shape[0], y_val.shape[1]))

# try without class_weight too
print("Training balanced svm_ovr...")
svm_ovr = multiclassSVM(X_train, y_train, decision_function_shape="ovr", class_weight="balanced")
dump(svm_ovr, "C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/svm/svm_ovr.joblib")
print("Done")
# svm_ovr.predict(X_val)

print("Training balanced svm_ovo..")
svm_ovo = multiclassSVM(X_train, y_train, decision_function_shape="ovo", class_weight="balanced")
dump(svm_ovo, "C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/svm/svm_ovo.joblib")
print("Done")
# svm_ovo.predict(X_val)

print("Training balanced linear svm...")
linear_svm = multiclassLinearSVM(X_train, y_train, class_weight="balanced")
dump(linear_svm, "C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/svm/linear_svm.joblib")
print("Done")
# linear_svm.predict(X_val)
'''
