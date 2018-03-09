'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#############################################
import common
import default_utils
from keras.callbacks import CSVLogger
from keras import backend as K

def initialize_parameters():
    mnist_common = common.MNIST(common.file_path,
        'mnist_default_model.txt',
        'keras',
        prog='mnist_mlp',
        desc='MNIST example'
    )

    # Initialize parameters
    gParameters = default_utils.initialize_parameters(mnist_common)
    csv_logger = CSVLogger('{}/params.log'.format(gParameters))

    return gParameters


def load_data(gParameters):
    return mnist.load_data()

def run(gParameters, data):

    batch_size = gParameters['batch_size']
    num_classes = 10
    epochs = gParameters['epochs']

    #############################################
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = data

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #############################################

def main():

    gParameters = initialize_parameters()
    data = load_data(gParameters)
    run(gParameters, data)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
