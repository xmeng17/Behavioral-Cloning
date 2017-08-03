from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda, Cropping2D, Convolution2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import csv


path_prefix = 'data/'
data_range = None
verbose = 1

batch_size = 64
epochs = 7
dropout_rate = 0.2
leaky=0.1


def get_train_valid_arr(path):
    arr = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            arr.append(line)
    arr = arr[:data_range]
    train_arr, valid_arr = train_test_split(arr, test_size=0.2)
    return train_arr, valid_arr

def generator(arr):
    num = len(arr)
    while True:
        shuffle(arr)
        for i in range(0, num, batch_size):
            batch_lines = arr[i:i + batch_size]

            images = []
            steerings = []
            for line in batch_lines:
                center_path = line[0]
                steering = float(line[3])
                center_real_path = path_prefix + center_path
                image = cv2.imread(center_real_path)
                #image = cv2.imread(center_path)
                images.append(image)
                steerings.append(steering)

                image_flip=np.fliplr(image)
                images.append(image_flip)
                steerings.append(-steering)

            X_train = np.array(images)
            y_train = np.array(steerings)

            yield shuffle(X_train, y_train)


def model(train_batch_num, valid_batch_num, train_generator, valid_generator):
    model = Sequential()
    model.add(Cropping2D(cropping=((45, 15), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    relu1=LeakyReLU(alpha=leaky)
    model.add(Convolution2D(16,(5,5),strides=(1,1),padding='valid'))
    model.add(relu1)
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    relu2 = LeakyReLU(alpha=leaky)
    model.add(Convolution2D(24, (5, 5), strides=(1, 1), padding='valid'))
    model.add(relu2)
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    relu3 = LeakyReLU(alpha=leaky)
    model.add(Convolution2D(32, (5, 5), strides=(1, 1), padding='valid'))
    model.add(relu3)
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(train_generator, steps_per_epoch=train_batch_num, epochs=epochs, verbose=verbose,
                              validation_data=valid_generator, validation_steps=valid_batch_num)
    model.save('model.h5')

train_arr, valid_arr = get_train_valid_arr(path_prefix + 'driving_log.csv')
train_batch_num = int(len(train_arr) / batch_size)
valid_batch_num = int(len(valid_arr) / batch_size)
train_generator = generator(train_arr)
valid_generator = generator(valid_arr)
model(train_batch_num, valid_batch_num, train_generator, valid_generator)
'''x_train,y_train=next(train_generator)
print(x_train[0])
print(y_train[0])'''
