from joblib import load
from prediction.predict import *

path_vgg_validation = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/VGG Features/valid/"
path_geometric_validation = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/Geometric Features/valid/"

path_results_v1 = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/predictions/vgg_sgd_classifier_balanced/predict_v1/"

sgd_class = load("C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/sgd/vgg_sgd_classifier_balanced.joblib")

predict_v1(sgd_class, path_vgg_validation, path_geometric_validation, path_results_v1, add_const_name=".csv",
           drop_const_name="_vgg.csv")
