import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Cropping2D, Lambda, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam

from utils import train, preprocess_img_size


"""
NVidia Model - see https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
NVidia uses 66x200 YUV image
We use 66x200 RGB images
"""
def create_image_branch(input_dim, conv_activation, fcn_activation, dropout_prob, use_bn):
    image_input = Input(shape=input_dim, name='image_input')

    x = image_input

    # Normalization - centered around zero with small standard deviation
    x = Lambda(lambda x: x / 255.0 - 0.5, name='image_normalization')(x)

    # CONV1, # filters = 24, kernel size = 5x5, out = 24@31x98
    x = conv_layer(x, 24, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV2, # filters = 36, kernel size = 5x5, out = 36@14x47
    x = conv_layer(x, 36, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV3, # filters = 48, kernel size = 5x5, out = 48@5x22
    x = conv_layer(x, 48, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV4, # filters = 64, kernel size = 3x3, out = 64@3x20
    x = conv_layer(x, 64, 3, 1, conv_activation, use_bn, dropout_prob)

    # CONV5, # filters = 64, kernel size = 3x3, out = 64@1x18
    x = conv_layer(x, 64, 3, 1, conv_activation, use_bn, dropout_prob)

    # Flatten layer
    x = Flatten()(x)

    # FCN1
    x = fc_layer(x, 1164, fcn_activation, use_bn, dropout_prob)

    # FCN2
    x = fc_layer(x, 100, fcn_activation, use_bn, dropout_prob)

    # FCN3
    x = fc_layer(x, 50, fcn_activation, use_bn, dropout_prob)

    # FCN4
    x = fc_layer(x, 10, fcn_activation, use_bn, dropout_prob)
    return image_input, x#, si, sb


"""
Builds the model which have 2 outputs: steering and speed.
"""
def complex_model(input_dim, conv_activation='relu', fcn_activation='relu', dropout_prob=0.5, use_bn=False):
    image_input, image_branch = create_image_branch(input_dim, conv_activation, fcn_activation, dropout_prob, use_bn)

    # Readout Layers
    steering_output = Dense(1, name='steering_output')(image_branch)
    speed_output = Dense(1, name='speed_output')(image_branch)

    model = Model(input=[image_input], output=[steering_output, speed_output])
    return model


"""
Build Convolutional layer [CONV - BN - ACTIVATION - MAX_POOL - DROPOUT] depending on passed params.
"""
def conv_layer(x, nb_filter, filter_size, stride, activation, use_bn, dropout_prob, pool_size=0, pool_stride=2):
    x = Convolution2D(nb_filter, filter_size, filter_size, subsample=(stride, stride))(x)

    if use_bn:
        x = BatchNormalization()(x)

    x = Activation(activation)(x)

    if pool_size > 0:
        x = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_stride, pool_stride), border_mode='valid')(x)

    if dropout_prob > 0:
        x = Dropout(dropout_prob)(x)
    return x

"""
Build FCN layer [FC - BN - ACTIVATION - DROPOUT] depending on passed params.
"""
def fc_layer(x, nb_hidden_units, activation, use_bn, dropout_prob):
    x = Dense(nb_hidden_units)(x)

    if use_bn:
        x = BatchNormalization()(x)

    x = Activation(activation)(x)

    if dropout_prob > 0:
        x = Dropout(dropout_prob)(x)
    return x


"""
see utils.generator
"""
def x_generator(images, speeds):
    return {'image_input': np.array(images)}

"""
see utils.generator
"""
def y_generator(steerings, speeds, throttles):
    return {'steering_output': np.array(steerings), 'speed_output': normalize_speeds(np.array(speeds))}


def get_croppping_dim():
    # crop from top and bottom
    # return (70, 20)
    return (40, 20)


def get_target_size():
    return (66, 200)


def predict(model, image):
    image = preprocess_img_size(image, get_croppping_dim(), get_target_size())
    pred = model.predict(image[None, :, :, :], batch_size=1)
    steering_angle = float(pred[0])
    speed = unnormalize_speed(float(pred[1]))
    return steering_angle, speed


"""
Convert speeds from the range [0, 30] to the range [-1, 1]
"""
def normalize_speeds(speeds):
    return speeds / 15. - 1.


"""
Convert speed from the range [-1, 1] to the range [0, 30]
"""
def unnormalize_speed(normalized_speed):
    return (normalized_speed + 1) * 15.


def main():
    data_dirs = [
        'data/udacity-origin-data/',

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

        # 'data/gathering3/track1-lap1/',
        # 'data/gathering3/track1-lap2-opposite/',
        # 'data/gathering3/track1-lap3-recovery/',
        # 'data/gathering3/track1-lap4/',
        # 'data/gathering3/track1-lap5-recovery/',
        # 'data/gathering3/track1-lap6-big/',
        # 'data/gathering3/track1-lap7-recovery/',

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
        'data/gathering6/track2-lap5-recovery',
    ]

    steering_correction = 0.2
    skip_steerings = [(0, 0.9)]#[(0, 0.98), (-1, 0.95), (1, 0.9)]
    lr = 1e-3
    activation = 'relu'
    dropout_prob = 0.2
    use_bn = False
    nb_epoch = 50
    use_side_cameras = False

    cropping_dim = get_croppping_dim()
    target_size = get_target_size()
    generator_batch_size = 128

    input_dim = (*target_size, 3)
    model = complex_model(input_dim, conv_activation=activation, fcn_activation=activation, dropout_prob=dropout_prob, use_bn=use_bn)
    print("Model: ")
    model.summary()
    model.compile(optimizer=Adam(lr=lr),
                  loss={'steering_output': 'mse', 'speed_output': 'mse'},
                  loss_weights={'steering_output': 1.0, 'speed_output': 1.0}
                  )

    train(model, 'models/2/model2.h5', x_generator, y_generator,
          data_dirs, skip_steerings, use_side_cameras, steering_correction,
          generator_batch_size, nb_epoch)


if __name__ == '__main__':
    main()