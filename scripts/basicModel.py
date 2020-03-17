import os
import numpy as np
import logging

from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate, SeparableConv2D, MaxPooling3D, Conv3D, DepthwiseConv2D
from tensorflow.keras.models import model_from_json

model_name = "VGG-like-50psize_with_SeparableConv2D_with_dropout"
logging.basicConfig(level=logging.INFO, filename=model_name + '.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

base_path = "/mood-detection/numpy_data_50_percent_size"
concentration_data_array = None
relaxed_data_array = None
logging.info("loading data from .npy files...")

for filename in os.listdir(base_path):
    filename = os.path.join(base_path, filename)
    if "concentration" in filename:
        logging.debug(filename)
        concentration_data_array = np.concatenate((concentration_data_array, np.load(filename))) if concentration_data_array is not None else np.load(filename)
    if "relaxed_array" in filename:
        logging.debug(filename)
        relaxed_data_array = np.concatenate((relaxed_data_array, np.load(filename))) if relaxed_data_array is not None else np.load(filename)
logging.info((concentration_data_array.shape, relaxed_data_array.shape))
logging.info((str(len(concentration_data_array)), str(len(relaxed_data_array))))
X = np.concatenate((relaxed_data_array, concentration_data_array))
Y = np.concatenate((np.zeros((len(relaxed_data_array),1)),np.ones((len(concentration_data_array),1))))
logging.info((X.shape, Y.shape))
relaxed_data_array, concentration_data_array = None, None
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X, Y = None, None
folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=1).split(X_train, y_train))

model_name = "VGG-like-50psize_with_SeparableConv2D"
basepath = os.path.join("/mood-detection", model_name)
if not os.path.exists(basepath):
    os.mkdir(basepath)

def get_model():
    model = Sequential()

    model.add(SeparableConv2D(input_shape=(110,170,42),filters=64,kernel_size=(3,3), padding="same", activation="relu",depth_multiplier=2))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    #model.add(Dropout(0.3))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    #model.add(Dropout(0.3))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    #model.add(Dropout(0.3))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    #model.add(Dropout(0.3))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.7))

    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))

    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=2, activation="softmax")) 

    adm = optimizers.Adam(lr=0.000001, decay=1e-6)
    model.compile(optimizer=adm, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #sgd = optimizers.SGD(lr=0.000001, momentum=0.7)
    #model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model

filepath= os.path.join(basepath, model_name + "-weights-improvement-{epoch:02d}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model = get_model()
for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = X_train[val_idx]
    y_valid_cv= y_train[val_idx]
    model.fit(X_train_cv, y_train_cv, validation_data=(X_valid_cv, y_valid_cv), epochs=10, batch_size=32, shuffle=True)
    score = model.evaluate(X_test,y_test)
    logging.info("Test set metrics after fold %s: %s: %.2f%%" % (str(j), model.metrics_names[1], score[1]*100))
model_json = model.to_json()
with open(os.path.join(basepath,model_name + ".json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(os.path.join(basepath,model_name + ".h5"))
logging.info("Saved model %s to disk" % (model_name))

