from joblib import load
from prediction.predict import *

path_hog_validation = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/HOG Features/valid/"
path_geometric_validation = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/Geometric Features/valid/"
path_hog_validation_label = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/valid/"

path_results_v1 = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/predictions/hog_sgd_classifier_balanced/predict_v1/"

sgd_class = load("C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/sgd/hog_sgd_classifier_balanced.joblib")

predict_v1(sgd_class, path_hog_validation, path_geometric_validation, path_results_v1)