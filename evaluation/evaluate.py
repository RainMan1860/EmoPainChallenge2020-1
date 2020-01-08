from evaluation.metrics import *
from training.preprocess import *

path_hog_validation_label = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/valid/"
path_results_v1 = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/predictions/vgg_sgd_classifier_balanced/predict_v1/"

y_true = concat_csv(path_hog_validation_label)
y_pred = concat_csv(path_results_v1)

# ccc = concordance_correlation_coefficient(y_true, y_pred)
# print("Concordance Correlation Coefficient: {}".format(ccc))

mae = MAE(y_true, y_pred)
print("Mean Absolute Error: {}".format(mae))

mse = MSE(y_true, y_pred)
print("Mean Squared Error: {}".format(mse))

rmse = RMSE(y_true, y_pred)
print("Root Mean Squared Error: {}".format(rmse))

pcc = pearson(y_true[0].values, y_pred[0].values)
print("Pearson Correlation Coefficient: {}".format(pcc[0]))
