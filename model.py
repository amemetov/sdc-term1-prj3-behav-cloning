import numpy as np

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers import Convolution2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam

from utils import train


"""
NVidia Model - see https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
NVidia uses 66x200 YUV image
We use 160x320 RGB images
"""
def nvidia_model(input_dim, cropping_dim, conv_activation='relu', fcn_activation='relu', dropout_prob=0.5, use_bn=False):
    model = Sequential()

    # Cropping Layer
    model.add(Cropping2D(cropping=(cropping_dim, (0, 0)), input_shape=input_dim))

    # Normalization - centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.0))

    # CONV1, # filters = 24, kernel size = 5x5 (24@31x158, origin: 24@31x98)
    add_conv_layer(model, 24, 5, 2, conv_activation, use_bn, dropout_prob, input_shape=input_dim)

    # CONV2, # filters = 36, kernel size = 5x5 (36@14x77, origin: 36@14x47)
    add_conv_layer(model, 36, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV3, # filters = 48, kernel size = 5x5 (48@5x37, origin: 48@5x22)
    add_conv_layer(model, 48, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV4, # filters = 64, kernel size = 3x3 (64@3x35, origin: 64@3x20)
    add_conv_layer(model, 64, 3, 1, conv_activation, use_bn, dropout_prob)

    # CONV5, # filters = 64, kernel size = 3x3 (64@1x33, origin: 64@1x18)
    add_conv_layer(model, 64, 3, 1, conv_activation, use_bn, dropout_prob)

    # Flatten layer
    model.add(Flatten())

    # FCN1
    add_fc_layer(model, 1164, fcn_activation, use_bn, dropout_prob)

    # FCN2
    add_fc_layer(model, 100, fcn_activation, use_bn, dropout_prob)

    # FCN3
    add_fc_layer(model, 50, fcn_activation, use_bn, dropout_prob)

    # FCN4
    add_fc_layer(model, 10, fcn_activation, use_bn, dropout_prob)

    # Readout Layer
    model.add(Dense(1))
    #model.add(Dense(1, activation='tanh'))

    return model

def add_conv_layer(model, nb_filter, filter_size, stride, activation, use_bn, dropout_prob, input_shape=None):
    if input_shape is None:
        model.add(Convolution2D(nb_filter, filter_size, filter_size, subsample=(stride, stride)))
    else:
        model.add(Convolution2D(nb_filter, filter_size, filter_size, subsample=(stride, stride), input_shape=input_shape))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(ELU())
    model.add(Dropout(dropout_prob))

def add_fc_layer(model, nb_hidden_units, activation, use_bn, dropout_prob):
    model.add(Dense(nb_hidden_units))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(ELU())
    model.add(Dropout(dropout_prob))


"""
see utils.generator
"""
def y_generator(steerings, throttles, speeds):
    return np.array(steerings)


def get_croppping_dim():
    # crop from top and bottom
    return (70, 20)


def get_target_size():
    origin_size = (160, 320)
    # cropping_dim = get_croppping_dim()
    # target_size = (origin_size[0] - cropping_dim[0] - cropping_dim[1], 320)  # origin target size

    # target_size = (66, 200) # nVidia model size

    target_size = origin_size
    return target_size


def main():
    data_dirs = ['data/udacity-origin-data/',
                 'data/track-1-lap1/',
                 'data/track-2-lap1/',
                 # 'data/track-2-lap2/',
                 # 'data/track-2-lap3/',
                 # 'data/track-2-recovery1/',
                 ]
    driving_log_file = 'driving_log.csv'

    steering_correction = 0.2
    lr = 1e-3
    activation = 'relu'
    dropout_prob = 0.2
    use_bn = False
    nb_epoch = 50

    cropping_dim = get_croppping_dim()
    target_size = get_target_size()
    generator_batch_size = 128

    input_dim = (*target_size, 3)
    model = nvidia_model(input_dim, cropping_dim, conv_activation=activation, fcn_activation=activation, dropout_prob=dropout_prob, use_bn=use_bn)
    print("Model: ")
    model.summary()
    model.compile(optimizer=Adam(lr=lr), loss='mse')

    train(model, 'models/1', y_generator,
          data_dirs, driving_log_file, steering_correction, cropping_dim, target_size,
          generator_batch_size, nb_epoch)


if __name__ == '__main__':
    main()


"""
This methods is used to find best hyperparams
"""
def grid_search(data_dirs, driving_log_file):
    angles = [0.1, 0.2, 0.3]
    use_grays = [True, False]
    lrs = [1e-4, 1e-3, 1e-2]
    activation = 'relu'
    dropout_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    use_bns = [True, False]
    nb_epoch = 10

    for angle in angles:
        for use_gray in use_grays:
            for lr in lrs:
                for dropout_prob in dropout_probs:
                    for use_bn in use_bns:
                        train(data_dirs, driving_log_file, angle, use_gray, lr, activation, dropout_prob, use_bn,
                              nb_epoch)