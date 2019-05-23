from keras.datasets import cifar10
import numpy as np

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
    # full_data = np.hstack((full_train_data, full_test_data))
    full_data = [np.concatenate((full_train_data[0], full_test_data[0]), axis=0),
                 np.concatenate((full_train_data[1], full_test_data[1]), axis=0)]
    X = full_data[0][st:end]
    y = full_data[1][st:end]
    return X, y

