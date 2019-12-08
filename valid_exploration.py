import os
import pandas as pd
import numpy as np


valid_label_folder = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/valid/"

num_valid_file = 0

for ft in os.listdir(valid_label_folder):
    num_valid_file += 1

print(num_valid_file)