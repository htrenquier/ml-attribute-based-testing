from keras.datasets import cifar10
import numpy as np
import attribute_analyser as aa
import math

def get_d_name(d_name):
    if d_name == 'cifar10':
        # Load CIFAR10 data
        train_data, test_data = cifar10.load_data()
        print(d_name + ' loaded.')
        input_shape = train_data[0].shape[1:]
        print(input_shape)
        val_data = test_data
    
    elif d_name == 'cifar10-2-5':
        train_data_orig, test_data_orig = cifar10.load_data()
        input_shape = train_data_orig[0].shape[1:]
        train_data = [train_data_orig[0][:20000], train_data_orig[1][:20000]]
        val_data = [train_data_orig[0][40000:], train_data_orig[1][40000:]]
        assert len(train_data[0]) == 20000 and len(val_data[0]) == 10000
        print(d_name + ' loaded.')
    elif d_name == 'cifar10-channelswitched':
        train_data_orig, test_data_orig = cifar10.load_data()
        input_shape = train_data_orig[0].shape[1:]
        train_imgs = []
        val_imgs = []
        for img in train_data_orig[0][:20000]:
            train_imgs.append(np.roll(img, 1, 2))
        for img in train_data_orig[0][40000:]:
            val_imgs.append(np.roll(img, 1, 2))
        train_data = [np.array(train_imgs), train_data_orig[1][:20000]]
        val_data = [np.array(val_imgs), train_data_orig[1][40000:]]
        assert len(train_data[0]) == 20000 and len(val_data[0]) == 10000
        print(d_name + ' loaded.')
    else:
        print('Not implemented')
        return
    return train_data, val_data


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
        cube = aa.ColorDensityCube(resolution=granularity)
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
        cube = aa.ColorDensityCube(resolution=granularity)
        cube.feed(image)
        c = cube.get_cube()
        argsmax = np.where(c == np.amax(c))
        image_cube[argsmax[0][0]][argsmax[1][0]][argsmax[2][0]].append(counter)
    return image_cube
