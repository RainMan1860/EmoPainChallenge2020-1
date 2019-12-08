import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



train_label_folder = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/valid/"

num_class = 11
num_train_file = 0
total_frame = 0
count_array = np.zeros(num_class, dtype=np.int64)

for ft in os.listdir(train_label_folder):
    num_train_file += 1
    data = pd.read_csv(train_label_folder + ft, header=None)
    total_frame += data.shape[0]
    class_count = data[0].value_counts()
    for index, value in class_count.items():
        count_array[index] += value

print("Suddivisione classi:")
for i in range(num_class):
    print("Classe {}: {}".format(i, count_array[i]))

print("Numero esempi di training: {}".format(num_train_file))
print("Numero totale di frame: {}".format(total_frame))


