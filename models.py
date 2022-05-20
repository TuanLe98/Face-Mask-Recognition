import os
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def count_image():
    img_train = list()
    img_test = list()
    for dirs in os.listdir("train"):
        count = len(os.listdir("train/" + dirs))
        img_train.append(count)
    for dirs in os.listdir("test"):
        count = len(os.listdir("test/" + dirs))
        img_test.append(count)
    return img_train, img_test


def model(data_train, data_test):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_initializer="he_uniform"))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=25)
    model.fit_generator(data_train, epochs=15, validation_data= data_test, callbacks=[callback])
    return model

def data_train():
    datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    data_train = datagen.flow_from_directory('train', target_size=(128, 128),
                                             batch_size=42,
                                             class_mode='binary')
    return data_train

def data_test():
    datagen = ImageDataGenerator(rescale=1/255)
    data_test = datagen.flow_from_directory('test', target_size=(128, 128),
                                            batch_size=42,
                                            class_mode='binary')
    return data_test
