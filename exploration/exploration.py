import os
import pandas as pd
import numpy as np


def explore(data_label_path, num_class):
    num_data_file = 0
    total_frame = 0
    count_array = np.zeros(num_class, dtype=np.int64)

    for ft in os.listdir(data_label_path):
        num_data_file += 1
        data = pd.read_csv(data_label_path + ft, header=None)
        total_frame += data.shape[0]
        class_count = data[0].value_counts()
        for index, value in class_count.items():
            count_array[index] += value

    return count_array, num_data_file, total_frame
