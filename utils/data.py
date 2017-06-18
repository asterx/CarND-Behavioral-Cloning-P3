# coding=utf-8
import numpy as np
import os, random, pandas, cv2
import skimage.transform as sktransform
import matplotlib.image as mpimg
from sklearn import model_selection
from pandas.core.frame import DataFrame

CAMERAS = [ 'left', 'center', 'right' ]
STEERING_CORRECTION = [ .25, 0., -.25 ] # Steering correction for all available cameras
IMAGE_SIZE = [ 100, 33 ] # Width, height
OFFSETS = [ .375, .125 ] # Top, bottom (percent)


def test_split(file_name, train_size):
    data = pandas.io.parsers.read_csv(file_name)
    data = balance(data)
    return model_selection.train_test_split(data, train_size = train_size)


def balance(data, bins = 1024, per_bin = 512):
    result = pandas.DataFrame()
    start = 0
    for end in np.linspace(0, 1, num = bins):
        df_range = data[(np.absolute(data.steering) >= start) & (np.absolute(data.steering) < end)]
        range_n = min(per_bin, df_range.shape[0])
        if range_n:
            result = pandas.concat([result, df_range.sample(range_n)])
        start = end
    return result


def get_random_camera_image(data, index, random_camera = True):
    camera = np.random.randint(len(CAMERAS)) if random_camera else 1
    image = mpimg.imread(os.path.join('./data', data[CAMERAS[camera]].values[index].strip()))
    angle = data.steering.values[index] + STEERING_CORRECTION[camera]
    return image, angle


def preprocess(image, top_offset=OFFSETS[0], bottom_offset=OFFSETS[1]):
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    return cv2.resize(image[top:-bottom, :], (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA)


def flip(x, y):
    flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
    x[flip_indices] = x[flip_indices, :, ::-1, :]
    y[flip_indices] = -y[flip_indices]
    return x, y


def augmentation(image):
    # Add random shadow as a vertical slice of image
    h, w = image.shape[0], image.shape[1]
    [ x1, x2 ] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1

    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)

    return image

#counter = 0

def batches(data, augment = True, batch_size = 128):
    #global counter

    while True:
        indices = np.random.permutation(data.count()[0])

        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, IMAGE_SIZE[1], IMAGE_SIZE[0], 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)

            # Read in and preprocess a batch of images
            for i in batch_indices:
                image, angle = get_random_camera_image(data, i)
                v_delta = 0

                #counter += 1
                #cv2.imwrite('debug/' + str(counter) + '_before.png', image)

                if augment:
                    v_delta = .08

                image = preprocess(
                    image,
                    top_offset = random.uniform(OFFSETS[0] - v_delta, OFFSETS[0] + v_delta),
                    bottom_offset = random.uniform(OFFSETS[1] - v_delta, OFFSETS[1] + v_delta)
                )

                #cv2.imwrite('debug/' + str(counter) + '_shift.png', image)

                if augment:
                    image = augmentation(image)

                #cv2.imwrite('debug/' + str(counter) + '_angle.png', image)

                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])

            # Randomly flip half of images in the batch
            x, y = flip(x, y)

            yield x, y
