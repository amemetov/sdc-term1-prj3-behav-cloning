import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Cropping2D, Lambda, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam

from utils import train, preprocess_img_size


def create_image_branch(input_dim, cropping_dim, conv_activation, fcn_activation, dropout_prob, use_bn):
    image_input = Input(shape=input_dim, name='image_input')

    # Cropping Layer100
    #x = Cropping2D(cropping=(cropping_dim, (0, 0)), input_shape=input_dim, name='cropping')(image_input)
    x = image_input

    # Normalization - centered around zero with small standard deviation
    x = Lambda(lambda x: x / 127.5 - 1.0, name='image_normalization')(x)

    # CONV1, # filters = 24, kernel size = 5x5 (24@31x158, origin: 24@31x98)
    x = conv_layer(x, 24, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV2, # filters = 36, kernel size = 5x5 (36@14x77, origin: 36@14x47)
    x = conv_layer(x, 36, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV3, # filters = 48, kernel size = 5x5 (48@5x37, origin: 48@5x22)
    x = conv_layer(x, 48, 5, 2, conv_activation, use_bn, dropout_prob)

    # CONV4, # filters = 64, kernel size = 3x3 (64@3x35, origin: 64@3x20)
    x = conv_layer(x, 64, 3, 1, conv_activation, use_bn, dropout_prob)

    # CONV5, # filters = 64, kernel size = 3x3 (64@1x33, origin: 64@1x18)
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



def complex_model(input_dim, cropping_dim, conv_activation='relu', fcn_activation='relu', dropout_prob=0.5, use_bn=False):
    image_input, image_branch = create_image_branch(input_dim, cropping_dim, conv_activation, fcn_activation, dropout_prob, use_bn)

    # Readout Layers
    steering_output = Dense(1, name='steering_output')(image_branch)
    speed_output = Dense(1, name='speed_output')(image_branch)

    model = Model(input=[image_input], output=[steering_output, speed_output])
    return model


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
    return {'steering_output': np.array(steerings), 'speed_output': np.array(speeds)/15. - 1.}


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
    speed = (float(pred[1]) + 1) * 15.
    return steering_angle, speed
    #return steering_angle, max(speed, 1.0)


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

    driving_log_file = 'driving_log.csv'

    steering_correction = 0.2
    skip_steerings = [(0, 0.9)]#[(0, 0.98), (-1, 0.95), (1, 0.9)]
    lr = 1e-3
    activation = 'relu'
    dropout_prob = 0.2
    use_bn = False
    nb_epoch = 50
    use_left_right = False

    cropping_dim = get_croppping_dim()
    target_size = get_target_size()
    generator_batch_size = 128

    input_dim = (*target_size, 3)
    model = complex_model(input_dim, cropping_dim, conv_activation=activation, fcn_activation=activation,
                         dropout_prob=dropout_prob, use_bn=use_bn)
    print("Model: ")
    model.summary()
    model.compile(optimizer=Adam(lr=lr),
                  loss={'steering_output': 'mse', 'speed_output': 'mse'},
                  loss_weights={'steering_output': 1.0, 'speed_output': 1.0}
                  )

    train(model, 'models/2/model2.h5', x_generator, y_generator,
          data_dirs, driving_log_file, steering_correction, use_left_right, skip_steerings,
          generator_batch_size, nb_epoch)


if __name__ == '__main__':
    main()