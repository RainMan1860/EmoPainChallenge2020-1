from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from training.nn.generator import *
import math
import matplotlib.pyplot as plt
from joblib import dump


# definition of the training and validation paths
path_vgg_train = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/vgg/train/"
path_vgg_train_label = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/labels/train/"

path_vgg_valid = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/vgg/valid/"
path_vgg_valid_label = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/labels/valid/"

# definition of constants
num_classes = 11
num_train_frames = 751586
num_valid_frames = 591990
batch_size = 64

# definition of neural network model
model = Sequential()
model.add(Dense(4096, input_dim=4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])

# callback to save the model after each epoch
filepath="C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/nn/vgg-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", save_best_only=False)

# start training
history = model.fit_generator(
    generatorCSVBatch(path_vgg_train, path_vgg_train_label, add_const="_vgg.csv", drop_const=".csv", ohe=num_classes,
                      batch_size=batch_size),
    steps_per_epoch=math.ceil(num_train_frames/batch_size),
    epochs=100,
    validation_data=generatorCSVBatch(path_vgg_valid, path_vgg_valid_label, add_const="_vgg.csv", drop_const=".csv",
                                      ohe=num_classes, batch_size=batch_size),
    validation_steps=math.ceil(num_valid_frames/batch_size),
    callbacks=[checkpoint]
)

print(history)
dump(history, "C:/Users/nmacchiarulo/PycharmProjects/EmoPainChallenge2020/models/nn/history-vgg.joblib")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()