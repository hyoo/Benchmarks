from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np

#############################################
import common
import default_utils
from keras.callbacks import CSVLogger
from keras import backend as K

def initialize_parameters():
    my_unet = common.UNET(common.file_path,
        'unet_configurable_model.txt',
        'keras',
        prog='unet_configurable',
        desc='UNET example'
    )

    # Initialize parameters
    gParameters = default_utils.initialize_parameters(my_unet)
    csv_logger = CSVLogger('{}/params.log'.format(gParameters))

    return gParameters


def load_data(gParameters):
    return common.load_data(gParameters)

def run(gParameters, data):

    #############################################
    model = build_model(gParameters, 512, 512)
    imgs_train, imgs_mask_train, imgs_test = data

    # train
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(imgs_train, imgs_mask_train, batch_size=gParameters['batch_size'], epochs=gParameters['epochs'], verbose=1, validation_split=0.2,
        shuffle=True, callbacks=[model_checkpoint])

    # predict & save result in npy format
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    np.save(gParameters['test_label_data'], imgs_mask_test)


def get_model(img_rows, img_cols):

    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = Concatenate(axis=3)([conv3,up7])

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10, name='unet')

    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = [dice_coef])

    return model

# custom metrics
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# custom loss
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def build_model(gParameters, img_rows, img_cols):
    # network params
    n_layers = gParameters['n_layers']
    filter_size = gParameters['filter_size']
    dropout = gParameters['dropout']
    activation = gParameters['activation']
    kernel_initializer = gParameters['kernel_initializer']

    inputs = Input((img_rows, img_cols, 1))
    conv_layers = []
    pool_layers = [inputs]

    for i in range(n_layers):
        conv = Conv2D(filter_size, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(pool_layers[i])
        conv = Conv2D(filter_size, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv)
        if dropout != None and i >= (n_layers - 2):
            conv = Dropout(dropout)(conv)
        pool = MaxPooling2D(pool_size = (2, 2))(conv)
        conv_layers.append(conv)
        pool_layers.append(pool)
        filter_size *= 2

    filter_size //= 4 # or filter_size = int(filter_size)

    for i in range(n_layers-1):
        up = Conv2D(filter_size, 2, activation = activation, padding = 'same')(UpSampling2D(size = (2,2))(conv_layers[-1]))
        merge = Concatenate(axis=3)([conv_layers[n_layers-i-2], up])

        conv = Conv2D(filter_size, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(merge)
        conv = Conv2D(filter_size, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv)
        conv_layers.append(conv)
        filter_size //= 2  # or filter_size = int(filter_size)

    # last layer
    last_conv = Conv2D(2, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv_layers[-1])
    last_conv = Conv2D(1, 1, activation = 'sigmoid')(last_conv)

    model = Model(inputs, last_conv)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()

    return model

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
