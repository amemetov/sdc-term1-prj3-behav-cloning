import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import skimage.transform

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

"""
This method loads all samples from passed dirs.
The method loads images from left and right cameras too to get more train data.
Steering angles for left and right cameras are adjusted using passed steering_correction value.
The result is the tuples of 2 numpy arrays, the first element of tuples is path to image,
the second element is the tuple (steering, throttle, speed)
"""
def load_samples(data_dirs, driving_log_file, steering_correction=0.2):
    X = []
    y = []

    for data_dir in data_dirs:
        print('Loading samples from dir {0}'.format(data_dir))
        driving_data = pd.DataFrame.from_csv(data_dir + driving_log_file, index_col=None)
        print('Origin Rows #: {0}'.format(driving_data.shape[0]))
        num_test_data = driving_data.shape[0]
        for row in range(num_test_data):
            steering = driving_data.get_value(row, 'steering')
            throttle = driving_data.get_value(row, 'throttle')
            speed = driving_data.get_value(row, 'speed')

            center = data_dir + driving_data.get_value(row, 'center')
            if os.path.isfile(center):
                X.append(center)
                y.append((steering, throttle, speed))

            left = data_dir + driving_data.get_value(row, 'left').strip()
            if os.path.isfile(left):
                X.append(left)
                y.append((steering - steering_correction, throttle, speed))

            right = data_dir + driving_data.get_value(row, 'right').strip()
            if os.path.isfile(right):
                X.append(right)
                y.append((steering + steering_correction, throttle, speed))

    return np.array(X), np.array(y)


def load_image(img_file):
    # load as RGB (openCV loads as BGR )
    return plt.imread(img_file)


"""
This methods generates data batches as numpy array of tuples (X, y).
The method loads image, pre-processes it, randomly augments the image, and then uses y_generator to prepare result y.
y_generator is the method which takes (steerings, throttles, speeds) and returns result y depending on the used model
"""
def generator(X, y, y_generator, cropping_dim=(70, 20), target_size=(66, 200), batch_size=32):
    num_samples = len(X)

    # Loop forever so the generator never terminates
    while 1:
        X, y = shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            X_batch = X[offset:offset + batch_size]
            y_batch = y[offset:offset + batch_size]

            images = []
            steerings = []
            throttles = []
            speeds = []
            for img_path, (steering, throttle, speed) in zip(X_batch, y_batch):
                img = load_image(img_path)
                img = preprocess_image(img, cropping_dim, target_size)
                img, steering, speed, throttle = random_augment(img, steering, throttle, speed)

                images.append(img)
                steerings.append(steering)
                throttles.append(throttles)
                speeds.append(speed)

            X_gen = np.array(images)
            y_gen = y_generator(steerings, throttles, speeds)
            yield X_gen, y_gen


"""
Pre-processing of the passed image.
preprocess_image2 contains steps which were tested (resizing, RGB2YUV and RGB2GRAS conversions),
but those steps did not give improvements,
so the current version does not contain any step.
Cropping and Normalization are moved to the Model.
"""
def preprocess_image(img, cropping_dim=(70, 20), target_size=(66, 200)):
    return img


def preprocess_image2(img, cropping_dim=(70, 20), target_size=(66, 200)):
    # crop image
    img = img[cropping_dim[0]:img.shape[0]-cropping_dim[1], :, :]

    # resize image
    if target_size[0] != img.shape[0] or target_size[1] != img.shape[1]:
        img = skimage.transform.resize(img, target_size, preserve_range=True)
        img = img.astype(np.ubyte) # do it for below RGB2YUV conversion, cause openCV cannot process float64 type

    # convert to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # if convert_to_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     img = np.reshape(img, (*img.shape, 1))

    # normalize
    img = img / 127.5 - 1.0

    return img


def random_augment(image, steering, throttle, speed):
    # 1. Flip
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        steering = -steering

    return image, steering, throttle, speed


def train(model, model_save_dir, y_generator,
          data_dirs, driving_log_file, steering_correction,
          cropping_dim=(70, 20), target_size=(66, 200),
          generator_batch_size=128, nb_epoch=10):
    print('========================================================')

    X_origin, y_origin = load_samples(data_dirs, driving_log_file, steering_correction=steering_correction)

    # shuffle data
    X_origin, y_origin = shuffle(X_origin, y_origin)

    # split data
    X_train, X_valid, y_train, y_valid = train_test_split(X_origin, y_origin, test_size=0.1)

    print('Train size: {0}'.format(X_train.shape[0]))
    print('Valid size: {0}'.format(X_valid.shape[0]))

    # compile and train the model using the generator function
    train_generator = generator(X_train, y_train, y_generator, cropping_dim=cropping_dim, target_size=target_size, batch_size=generator_batch_size)
    validation_generator = generator(X_valid, y_valid, y_generator, cropping_dim=cropping_dim, target_size=target_size, batch_size=generator_batch_size)

    checkpoint = ModelCheckpoint(model_save_dir + '/model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    # tensor_board = TensorBoard(log_dir='./tb-logs', histogram_freq=1, write_graph=True, write_images=True)

    nb_samples_per_epoch = len(X_train)
    nb_valid_samples = len(X_valid)

    history = model.fit_generator(train_generator, samples_per_epoch=nb_samples_per_epoch,
                                  validation_data=validation_generator, nb_val_samples=nb_valid_samples,
                                  callbacks=[checkpoint, early_stopping],  # , tensor_board],
                                  nb_epoch=nb_epoch,
                                  )

    plot_history_curve(history)

    return model, history


def plot_history_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./loss-curve.png")
    plt.close()

def check_balance(dataset_labels, dataset_name):
    classes = np.unique(dataset_labels)
    num_classes = classes.size
    items_per_classes = []

    # for c in range(num_classes):
    for c in classes:
        items_per_classes.append(len(dataset_labels[dataset_labels == c]))

    # print(items_per_classes)

    plt.figure()
    plt.bar(np.arange(num_classes), items_per_classes)
    max_items_per_classes = max(items_per_classes)
    # plt.axis([0, num_classes, 0, 1.1 * max_items_per_classes])
    # plt.xticks(classes)

    plt.title("Number of samples for each class for {0}. Total #: {1}".format(dataset_name, len(dataset_labels)))
    plt.grid(True)
