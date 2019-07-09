from keras.datasets import cifar10
import numpy as np
import metrics
import metrics_color
from keras.preprocessing import image
from keras import utils

def format_data(data, num_classes):
    """
    Formats image data for model predicting
    :param data: (data, label) format
    :param num_classes: int
    :return: (data, label) float32 formated images data
    """
    (x, y) = data
    x = x.astype('float32')
    x /= 255
    y = utils.to_categorical(y, num_classes)
    return x, y


def predict_and_acc(model, test_data):
    """
    Predicts and computes accuracy of a model.
    :param model:
    :param test_data: Unformatted test data
    :return: float Accuracy, List[int] Predicted classes, List[List[float]] List of predictions
    """
    test_data_f = format_data(test_data, 10)
    y_predicted = model.predict(test_data_f[0])
    predicted_classes = np.argmax(y_predicted, axis=1)
    true_classes = np.argmax(test_data_f[1], axis=1)
    acc = metrics.accuracy(predicted_classes, true_classes)
    return acc, predicted_classes, y_predicted


def predict_batch(model, images):
    """
    Returns class predictions for image batch
    :param model:
    :param images:
    :return: List[int] of predicted classes
    """
    if images:
        y_predicted = model.predict(images)
        predicted_classes = np.argmax(y_predicted, axis=1)
        return predicted_classes.tolist()
    else:
        return []


def predict_dataset(filenames, path, model, model_preprocess_function):
    """
    For predicting large amount of images (e.g. imagenet)
    :param filenames: file of filenames
    :param model_preprocess_function:
    :param path: path of test images
    :param model:
    :return: predictions
    """
    y_predicted = []
    batch_size = 32
    batch = []
    for filename in filenames:
        batch.append(preprocess(path+filename, model_preprocess_function))
        if len(batch) >= batch_size:
            y_predicted = y_predicted + model.predict(np.array(batch)).tolist()
            batch = []
    y_predicted = y_predicted + model.predict(np.array(batch)).tolist()
    return y_predicted


def preprocess(file_path, model_preprocess_function):
    """
    Image pre-processing for large dataset prediction
    :param file_path:
    :param model_preprocess_function:
    :return:
    """
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    x = model_preprocess_function(x)
    return x


def train_val_split(d_name, (train_start, train_end), (val_start, val_end)):
    if d_name == 'cifar10':
        full_train_data, full_test_data = cifar10.load_data()
    else:
        return
    full_data = np.hstack((full_train_data, full_test_data))
    train_data = full_data[train_start, train_end]
    val_data = full_data[val_start, val_end]
    return train_data, val_data


def get_data(d_name, (st, end)):
    if d_name == 'cifar10':
        full_train_data, full_test_data = cifar10.load_data()
    else:
        return
    X = np.concatenate((full_train_data[0], full_test_data[0]), axis=0)[st:end]
    y = np.concatenate((full_train_data[1], full_test_data[1]), axis=0)[st:end]
    return X, y


def cifar10_color_domains(granularity, frequence, data_range=(50000, 60000)):
    assert granularity and not granularity & (granularity - 1)  # granularity is power of 2
    image_cube = [[[[] for _ in xrange(granularity)]
                   for _ in xrange(granularity)]
                  for _ in xrange(granularity)]
    data_orig = get_data('cifar10', data_range)
    n_pix = 32*32
    for counter, image in enumerate(data_orig[0]):
        cube = metrics_color.ColorDensityCube(resolution=granularity)
        cube.feed(image)
        for i in xrange(granularity):
            for j in xrange(granularity):
                for k in xrange(granularity):
                    if cube[i, j, k]/n_pix >= frequence:
                        image_cube[i][j][k].append(counter)
    return image_cube


def cifar10_maxcolor_domains(granularity, data_range=(50000, 60000)):
    assert granularity and not granularity & (granularity - 1)  # granularity is power of 2
    image_cube = [[[[] for _ in xrange(granularity)]
                   for _ in xrange(granularity)]
                  for _ in xrange(granularity)]
    data_orig = get_data('cifar10', data_range)
    for counter, image in enumerate(data_orig[0]):
        cube = metrics_color.ColorDensityCube(resolution=granularity)
        cube.feed(image)
        c = cube.get_cube()
        argsmax = np.where(c == np.amax(c))
        image_cube[argsmax[0][0]][argsmax[1][0]][argsmax[2][0]].append(counter)
    return image_cube


def cifar10_nth_maxcolor_domains(granularity, n, data_range=(50000, 60000)):
    assert granularity and not granularity & (granularity - 1)  # granularity is power of 2
    image_cube = [[[[] for _ in xrange(granularity)]
                   for _ in xrange(granularity)]
                  for _ in xrange(granularity)]
    data_orig = get_data('cifar10', data_range)
    for counter, image in enumerate(data_orig[0]):
        cube = metrics_color.ColorDensityCube(resolution=granularity)
        cube.feed(image)
        c = cube.get_cube()
        argsmax = np.where(c == np.amax(c))
        for k in xrange(n-1):
            c[argsmax[0][0]][argsmax[1][0]][argsmax[2][0]] = 0
            argsmax = np.where(c == np.amax(c))
        image_cube[argsmax[0][0]][argsmax[1][0]][argsmax[2][0]].append(counter)
    return image_cube


def cube_cardinals(cube):
    g = len(cube)
    for i in xrange(g):
        for j in xrange(g):
            for k in xrange(g):
                cube[i][j][k] = len(cube[i][j][k])
    return cube


def print_ds_color_distrib():
    print("Max colors")
    g = 4
    ds_range = (0, 60000)
    max1 = cifar10_maxcolor_domains(g, ds_range)
    max1 = cube_cardinals(max1)
    cube_max1 = metrics_color.ColorDensityCube(g, max1)
    cube_max1.normalize()
    cube_max1.plot_cube(title='Max color distribution')
    print('max1 plotted')
    max2 = cifar10_nth_maxcolor_domains(g, 2, ds_range)
    max2 = cube_cardinals(max2)
    cube_max2 = metrics_color.ColorDensityCube(g, max2)
    cube_max2.normalize()
    cube_max2.plot_cube(title='2nd Max color distribution')
