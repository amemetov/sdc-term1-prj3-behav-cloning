import numpy as np
import pandas as pd
import cv2
import os
import shutil
import matplotlib.pyplot as plt
import skimage.transform

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

"""
This method loads all samples from passed dirs.
The method loads images from left and right cameras too to get more train data.
Steering angles for left and right cameras are adjusted using passed steering_correction value.
The result is the list of tuples (image_path, steering, throttle, brake, speed)
"""
def load_samples(data_dirs, driving_log_file, steering_correction=0.2, use_left_right=False, skip_steerings=None):
    samples = []

    for data_dir in data_dirs:
        print('Loading samples from dir {0}'.format(data_dir))
        driving_data = pd.DataFrame.from_csv(os.path.join(data_dir, driving_log_file), index_col=None)
        print('Origin Rows #: {0}'.format(driving_data.shape[0]))
        num_test_data = driving_data.shape[0]

        is_recovery = data_dir.endswith("recovery") or data_dir.endswith("recovery/")

        for row in range(num_test_data):
            steering = float(driving_data.get_value(row, 'steering'))
            throttle = float(driving_data.get_value(row, 'throttle'))
            brake = float(driving_data.get_value(row, 'brake'))
            speed = float(driving_data.get_value(row, 'speed'))

            # combine throttle with brake
            throttle = throttle - brake


            if not is_recovery and speed < 5:
                # remove samples with small speed
                continue


            if is_recovery:
                # for recovery mode remove sample with small steering
                if abs(steering) < 0.5:
                    continue

                # increase speed for recovery samples
                if speed < 10:
                    speed = np.random.uniform(10, 15)


            skip = False
            if skip_steerings is not None and isinstance(skip_steerings, list):
                for skip_steering, skip_rand in skip_steerings:
                    if steering == skip_steering and np.random.random() < skip_rand:
                        skip = True
                        break

            if skip:
                continue

            center = os.path.join(data_dir, driving_data.get_value(row, 'center'))
            samples.append((center, steering, throttle, speed))

            if use_left_right and steering != 0:
                left_speed = right_speed = speed
                left_steering = np.clip(steering + steering_correction, -1, 1)
                right_steering = np.clip(steering - steering_correction, -1, 1)

                left = os.path.join(data_dir, driving_data.get_value(row, 'left').strip())
                samples.append((left, left_steering, throttle, left_speed))

                right = os.path.join(data_dir, driving_data.get_value(row, 'right').strip())
                samples.append((right, right_steering, throttle, right_speed))

    return samples


def load_image(img_file):
    # load as RGB (openCV loads as BGR )
    return plt.imread(img_file)


def save_image(image, path, img_format=None):
    plt.imsave(path, image, format=img_format)

"""
This methods generates data batches as numpy array of tuples (X, y).
samples is array of tuples (image_path, steering, throttle, speed)
The method loads image, pre-processes it, randomly augments the image, and then uses y_generator to prepare result y.
y_generator is the method which takes (steerings, speeds, throttles) and returns result y depending on the used model
"""
def generator(samples, x_generator, y_generator, batch_size=32):
    num_samples = len(samples)

    # Loop forever so the generator never terminates
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            samples_batch = samples[offset:offset + batch_size]

            images = []
            steerings = []
            throttles = []
            speeds = []
            for (img_path, steering, throttle, speed) in samples_batch:
                img = load_image(img_path)
                img = preprocess_img_size(img)
                img, steering, throttle, speed = randomomize_image(img, steering, throttle, speed)

                images.append(img)
                steerings.append(steering)
                throttles.append(throttle)
                speeds.append(speed)

            X_batch = x_generator(images, speeds)
            y_batch = y_generator(steerings, speeds, throttles)
            yield X_batch, y_batch


def preprocess_img_size(img, cropping_dim=(40, 20), target_size=(66, 200)):
    # crop image
    img = img[cropping_dim[0]:img.shape[0] - cropping_dim[1], :, :]

    # resize image
    if target_size[0] != img.shape[0] or target_size[1] != img.shape[1]:
        img = skimage.transform.resize(img, target_size, preserve_range=True)
        img = img.astype(np.ubyte)

    return img


"""
Pre-processing of the passed image.
preprocess_image2 contains steps which were tested (RGB2YUV and RGB2GRAS conversions),
but those steps did not give improvements,
so the current version does not contain any step.
Cropping and Normalization are moved to the Model.
"""
def preprocess_image(img):
    return img


def preprocess_image2(img):
    # convert to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    #if convert_to_gray:
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img = np.reshape(img, (*img.shape, 1))

    # normalize
    img = img / 127.5 - 1.0

    return img


def randomomize_image(image, steering, throttle, speed):
    if np.random.random() < 0.5:
        image, steering = flip_image(image, steering)

    # if np.random.random() < 0.5:
    #    image = randomize_brightness(image)
    #
    # if np.random.random() < 0.5:
    #     image = randomize_saturation(image)
    #
    # if np.random.random() < 0.5:
    #     image = randomize_noise(image)

    return image, steering, throttle, speed


def flip_image(image, steering):
    return np.fliplr(image), -steering


def randomize_noise(img):
    noisy = img + 50*np.random.random(img.shape)
    noisy = noisy.astype(np.ubyte)
    noisy[:, :, :] = np.clip(noisy[:, :, :], 0, 255)
    return noisy


def randomize_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand_val = np.random.uniform(0.2, 2.5)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2]*rand_val, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def randomize_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand_val = np.random.uniform(0.2, 10.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rand_val, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def train(model, model_save_path, x_generator, y_generator,
          data_dirs, driving_log_file,
          steering_correction, use_left_right=False, skip_steerings=None,
          generator_batch_size=128, nb_epoch=10):
    print('========================================================')

    origin_samples = load_samples(data_dirs, driving_log_file, steering_correction=steering_correction, use_left_right=use_left_right, skip_steerings=skip_steerings)

    # shuffle data
    origin_samples = shuffle(origin_samples)

    # split data
    train_samples, valid_samples = train_test_split(origin_samples, test_size=0.1)

    print('Train size: {0}'.format(len(train_samples)))
    print('Valid size: {0}'.format(len(valid_samples)))

    # compile and train the model using the generator function
    train_generator = generator(train_samples, x_generator, y_generator, batch_size=generator_batch_size)
    validation_generator = generator(valid_samples, x_generator, y_generator, batch_size=generator_batch_size)

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    # tensor_board = TensorBoard(log_dir='./tb-logs', histogram_freq=1, write_graph=True, write_images=True)

    nb_samples_per_epoch = len(train_samples)
    nb_valid_samples = len(valid_samples)

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


def preprocess_img_data(src_data_dir, dst_data_dir,
                        cropping_dim, target_size,
                        img_sub_dir='IMG', csv_file_name='driving_log.csv'):
    if not os.path.isdir(src_data_dir):
        print("Dir {0} either does not exist or is not dir".format(src_data_dir))
        return

    if os.path.exists(dst_data_dir):
        print("Dir {0} already exists".format(dst_data_dir))
        return

    src_img_dir = os.path.join(src_data_dir, img_sub_dir)
    if not os.path.isdir(src_img_dir):
        print("Dir {0} either does not exist or is not dir".format(src_img_dir))
        return

    src_csv_file = os.path.join(src_data_dir, csv_file_name)
    if not os.path.isfile(src_csv_file):
        print("File {0} either does not exist or is not file".format(src_csv_file))
        return

    os.mkdir(dst_data_dir)
    print("Created dir {0}".format(dst_data_dir))

    # 1 Copy csv file
    dst_csv_file = os.path.join(dst_data_dir, csv_file_name)
    shutil.copyfile(src_csv_file, dst_csv_file)
    print("File {0} copied to file {1}".format(src_csv_file, dst_csv_file))

    # create IMG dir
    dst_img_dir = os.path.join(dst_data_dir, img_sub_dir)
    os.mkdir(dst_img_dir)
    print("Created dir {0}".format(dst_img_dir))

    # created preprocessed images
    image_list = os.listdir(src_img_dir)
    for src_path in image_list:
        src_img = load_image(os.path.join(src_img_dir, src_path))
        dst_img = preprocess_image(src_img, cropping_dim=cropping_dim, target_size=target_size)
        dst_path = os.path.join(dst_img_dir, src_path)
        save_image(dst_img, dst_path, img_format='jpeg')

    print('Done. Preprocessed {0} images'.format(len(image_list)))


