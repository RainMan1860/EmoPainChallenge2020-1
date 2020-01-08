from joblib import dump
from training.train import *

path_vgg_train = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/vgg/train/"
path_vgg_train_label = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/labels/train/"

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

print("Training the SGD Classifier...")
sgd_class = SGD_classifier(path_vgg_train, path_vgg_train_label, add_const_name="_vgg.csv", drop_const_name=".csv",
                           max_iter=1000, class_weight="balanced", classes=classes)
dump(sgd_class, "C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/sgd/vgg_sgd_classifier_balanced.joblib")
print("Done")
