import numpy as np

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam

from utils import train


"""
NVidia Model - see https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
NVidia uses 66x200 YUV image
We use 66x200 RGB images
"""
def nvidia_model(input_dim, conv_activation='relu', fcn_activation='relu', dropout_prob=0.5, use_bn=False):
    model = Sequential()

    # Normalization - centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_dim))

    # CONV1, # filters = 24, kernel size = 5x5, out = 24@31x98
    add_conv_layer(model, 24, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV2, # filters = 36, kernel size = 5x5, out = 36@14x47
    add_conv_layer(model, 36, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV3, # filters = 48, kernel size = 5x5, out = 48@5x22
    add_conv_layer(model, 48, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV4, # filters = 64, kernel size = 3x3, out = 64@3x20
    add_conv_layer(model, 64, 3, 1, conv_activation, use_bn, dropout_prob)

    # CONV5, # filters = 64, kernel size = 3x3, out = 64@1x18
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

    return model


def add_conv_layer(model, nb_filter, filter_size, stride, activation, use_bn, dropout_prob, pool_size=0, pool_stride=2):
    model.add(Convolution2D(nb_filter, filter_size, filter_size, subsample=(stride, stride)))

    if use_bn:
        model.add(BatchNormalization())

    model.add(Activation(activation))
    #model.add(ELU())

    if pool_size > 0:
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_stride, pool_stride), border_mode='valid'))

    model.add(Dropout(dropout_prob))

def add_fc_layer(model, nb_hidden_units, activation, use_bn, dropout_prob):
    model.add(Dense(nb_hidden_units))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    #model.add(ELU())
    model.add(Dropout(dropout_prob))


"""
see utils.generator
"""
def x_generator(images, speeds):
    return np.array(images)

"""
see utils.generator
"""
def y_generator(steerings, speeds, throttles):
    return np.array(steerings)


def get_croppping_dim():
    # crop from top and bottom
    # return (70, 20)
    return (40, 20)


def get_target_size():
    return (66, 200)


def main():
    data_dirs = [
        # 'data/udacity-origin-data/',

        # 'data/gathering1/track1-lap1/',
        # 'data/gathering1/track1-lap2-throttle/',
        # 'data/gathering1/track2-lap1/',
        # 'data/gathering1/track2-lap2/',
        # 'data/gathering1/track2-lap3/',
        # 'data/gathering1/track2-lap4-throttle/',
        # 'data/gathering1/track2-recovery1/',
        #
        # 'data/gathering2/track1-lap1/',
        # 'data/gathering2/track2-lap1/',

        'data/gathering3/track1-lap1/',
        'data/gathering3/track1-lap2-opposite/',
        'data/gathering3/track1-lap3-recovery/',
        'data/gathering3/track1-lap4/',
        'data/gathering3/track1-lap5-recovery/',
        'data/gathering3/track1-lap6-big/',
        'data/gathering3/track1-lap7-recovery/',

        'data/gathering4/track1-lap1',
        'data/gathering4/track1-lap2-recovery',
        'data/gathering4/track1-lap3-opposite',
        'data/gathering4/track2-lap1',
        'data/gathering4/track2-lap2-recovery',
        'data/gathering4/track2-lap3-recovery',

        'data/gathering5/track1-lap1',
        'data/gathering5/track2-lap1',

        'data/gathering6/track2-lap1',
        'data/gathering6/track2-lap2-opposite',
        'data/gathering6/track2-lap3',
        'data/gathering6/track2-lap4-recovery',
    ]



    skip_steerings = [(0, 0.9)] #[(0, 0.98), (-1, 0.95), (1, 0.9)]
    use_side_cameras = False #True
    steering_correction = 0.25

    cropping_dim = get_croppping_dim()
    target_size = get_target_size()
    input_dim = (*target_size, 3)

    lr = 1e-3
    activation = 'relu'
    dropout_prob = 0.2
    use_bn = False
    nb_epoch = 50
    generator_batch_size = 128

    model = nvidia_model(input_dim, conv_activation=activation, fcn_activation=activation, dropout_prob=dropout_prob, use_bn=use_bn)
    print("Model: ")
    model.summary()
    model.compile(optimizer=Adam(lr=lr), loss='mse')

    train(model, 'models/1/model.h5', x_generator, y_generator,
          data_dirs, skip_steerings, use_side_cameras, steering_correction,
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


"""
My model from Project2 TrafficSign Classifier.
For Project3 gives poor result.
"""
def my_model(input_dim, cropping_dim, conv_activation, fcn_activation, dropout_prob, use_bn):
    model = Sequential()

    # Normalization - centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.0, name='image_normalization', input_shape=input_dim))

    # CONV1
    add_conv_layer(model, 32, 7, 1, conv_activation, use_bn, dropout_prob, pool_size=0)
    # CONV2
    add_conv_layer(model, 32, 7, 1, conv_activation, use_bn, dropout_prob, pool_size=2)
    # CONV3
    add_conv_layer(model, 64, 5, 1, conv_activation, use_bn, dropout_prob, pool_size=0)
    # CONV4
    add_conv_layer(model, 64, 5, 1, conv_activation, use_bn, dropout_prob, pool_size=2)

    # Flatten layer
    model.add(Flatten())

    add_fc_layer(model, 256, fcn_activation, use_bn, dropout_prob)
    add_fc_layer(model, 256, fcn_activation, use_bn, dropout_prob)
    add_fc_layer(model, 128, fcn_activation, use_bn, dropout_prob)
    add_fc_layer(model, 64, fcn_activation, use_bn, dropout_prob)

    model.add(Dense(1))

    return model

