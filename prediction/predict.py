import os
import pandas as pd
import numpy as np


# predict_v1 manages the success=0 frames in this way:
# if success=0
#   classify the frame as 0 (the majority of the classes)
# else
#   classify the features with the model trained
# The predictions are saved in a csv file
def predict_v1(model, X_val_path, success_val_path, result_path, add_const_name="", drop_const_name=""):
    predictions = []
    for df_path in os.listdir(X_val_path):
        print("Predicting {}".format(df_path))
        X = pd.read_csv(X_val_path + df_path, header=None)
        X_geometric = pd.read_csv(success_val_path + df_path.replace(drop_const_name, add_const_name))

        # for row in X.itertuples(index=False):
        for i, row in X_geometric.iterrows():
            if row[4] == 0:
                predictions.append(0)
            else:
                # pred = model.predict(np.array(row).reshape(1, -1))
                sample = X.iloc[i].values
                pred = model.predict(sample.reshape(1, -1))
                predictions.append(pred[0])

        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(result_path + df_path, header=False, index=False)
        print("Predictions for {}: {}".format(df_path, len(predictions)))
        predictions = []