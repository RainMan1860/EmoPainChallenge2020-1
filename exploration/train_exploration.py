import exploration as exp
import matplotlib.pyplot as plt


# train_label_folder = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/train/"
# path of the cleaned training set
train_label_folder = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/labels/train/"

num_class = 11

count_array, num_train_file, total_frame = exp.explore(train_label_folder, num_class)

print("Suddivisione classi:")
for i in range(num_class):
    print("Classe {}: {}".format(i, count_array[i]))

print("Numero esempi di training: {}".format(num_train_file))
print("Numero totale di frame: {}".format(total_frame))

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# classes = range(0, num_class)
# ax.bar(classes, count_array)
# plt.show()
