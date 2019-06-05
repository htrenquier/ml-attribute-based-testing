import keras.applications as kapp
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
import os
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D

import numpy as np

def model_param(model_type):
    # model_structure, batch_size
    return {
        'densenet121': (32,
                        'categorical_crossentropy',
                        'adam',
                        ['accuracy']),
        'densenet169': (32,
                        'categorical_crossentropy',
                        'adam',
                        ['accuracy']),
        'densenet201': (32,
                        'categorical_crossentropy',
                        'adam',
                        ['accuracy']),
        'mobilenet': (32,
                        'categorical_crossentropy',
                        'adam',
                        ['accuracy']),
        'mobilenetv2': (32,
                        'categorical_crossentropy',
                        'adam',
                        ['accuracy']),
        'nasnet': (32,
                    'categorical_crossentropy',
                    'adam',
                    ['accuracy']),
        'resnet50': (32,
                    'categorical_crossentropy',
                    'adam',
                    ['accuracy']),
        'vgg16': (32,
                    'categorical_crossentropy',
                    'adam',
                    ['accuracy']),
        'vgg19': (32,
                    'categorical_crossentropy',
                    'adam',
                    ['accuracy']),
    }[model_type]


def model_struct(model_type, input_shape, classes, weights=None):
    # model_structure, batch_size
    if model_type == 'densenet121':
        return kapp.densenet.DenseNet121(include_top=True,
                                         weights=weights,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         pooling=None,
                                         classes=classes)
    elif model_type == 'densenet169':
        return kapp.densenet.DenseNet169(include_top=True,
                                         weights=weights,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         pooling=None,
                                         classes=classes)
    elif model_type == 'densenet201':
        return kapp.densenet.DenseNet201(include_top=True,
                                         weights=weights,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         pooling=None,
                                         classes=classes)
    elif model_type == 'mobilenet':
        return kapp.mobilenet.MobileNet(include_top=True,
                                        weights=weights,
                                        input_tensor=None,
                                        input_shape=input_shape,
                                        pooling=None,
                                        classes=classes)
    elif model_type == 'mobilenetv2':
        return kapp.mobilenet_v2.MobileNetV2(include_top=True,
                                             weights=weights,
                                             input_tensor=None,
                                             input_shape=input_shape,
                                             pooling=None,
                                             classes=classes)
    elif model_type == 'nasnet':
        return kapp.nasnet.NASNetMobile(include_top=True,
                                        weights=weights,
                                        input_tensor=None,
                                        input_shape=input_shape,
                                        pooling=None,
                                        classes=classes)
    elif model_type == 'resnet50':
        return kapp.resnet50.ResNet50(include_top=True,
                                      weights=weights,
                                      input_tensor=None,
                                      input_shape=input_shape,
                                      pooling=None,
                                      classes=classes)
    elif model_type == 'vgg16':
        return kapp.vgg16.VGG16(include_top=True,
                                weights=weights,
                                input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif model_type == 'vgg19':
        return kapp.vgg19.VGG19(include_top=True,
                                weights=weights,
                                input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)


def load_imagenet_model(model_type):
    if model_type == 'densenet121':
        return kapp.densenet.DenseNet121(), kapp.densenet.preprocess_input
    elif model_type == 'densenet169':
        return kapp.densenet.DenseNet169(), kapp.densenet.preprocess_input
    elif model_type == 'densenet201':
        return kapp.densenet.DenseNet201(), kapp.densenet.preprocess_input
    elif model_type == 'mobilenet':
        return kapp.mobilenet.MobileNet(), kapp.mobilenet.preprocess_input
    elif model_type == 'mobilenetv2':
        return kapp.mobilenet_v2.MobileNetV2(), kapp.mobilenet_v2.preprocess_input
    elif model_type == 'nasnet':
        return kapp.nasnet.NASNetMobile(), kapp.nasnet.preprocess_input
    elif model_type == 'resnet50':
        return kapp.resnet50.ResNet50(), kapp.resnet50.preprocess_input
    elif model_type == 'vgg16':
        return kapp.vgg16.VGG16(), kapp.vgg16.preprocess_input
    elif model_type == 'vgg19':
        return kapp.vgg19.VGG19(), kapp.vgg19.preprocess_input


# def format_data(train_data, test_data, num_classes):
#     (x_train, y_train), (x_test, y_test) = train_data, test_data
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255
#     y_train = utils.to_categorical(y_train, num_classes)
#     y_test = utils.to_categorical(y_test, num_classes)
#     return (x_train, y_train), (x_test, y_test)


def format_data(data, num_classes):
    (x, y) = data
    x = x.astype('float32')
    x /= 255
    y = utils.to_categorical(y, num_classes)
    return x, y


# def select_data(dataset_name,ratio):
#     nb_train, nb_val, nb_test = ratio

def train_and_save(model, epochs, data_augmentation, weight_file, train_data, val_data, batch_size, regression=False):

    (x_train, y_train) = format_data(train_data, 10)
    (x_val, y_val) = format_data(val_data, 10)

    if regression:
        # For regression
        y_val = val_data[1]
        y_train = train_data[1]

    checkpoint = ModelCheckpoint(
        weight_file,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto'
    )

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=0,
            shuffle=True,
            callbacks=[checkpoint]
        )
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by dataset std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in 0 to 180 degrees
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val),
            workers=4,
            verbose=2,
            steps_per_epoch=(50000 / batch_size),
            callbacks=[checkpoint]
        )

    # score = model.evaluate(x_val, y_val, verbose=0)
    # print('Test loss:', score[0])
    # print('Val accuracy:', score[1])
    # model.save_weights(weight_file)


def train(model_type, dataset, epochs, data_augmentation, path=''):

    if data_augmentation is True:
        # With DataAugmentation
        model_name = '%s_%s_%dep_wda' % (model_type, dataset, epochs)
    else:
        # WithOut DataAugmentation
        model_name = '%s_%s_%dep_woda' % (model_type, dataset, epochs)

    print('###---> ' + model_name + ' <---###')
    weight_file = model_name + '.h5'

    if dataset == 'cifar10':
        # Load CIFAR10 data
        train_data, test_data = cifar10.load_data()
        print(dataset + ' loaded.')
        input_shape = train_data[0].shape[1:]
        print(input_shape)
        model = model_struct(model_type, input_shape, 10)
        print(model_type + ' structure loaded.')
        val_data = test_data

    elif dataset == 'cifar10-2-5':
        train_data_orig, test_data_orig = cifar10.load_data()
        input_shape = train_data_orig[0].shape[1:]
        train_data = [train_data_orig[0][:20000], train_data_orig[1][:20000]]
        val_data = [train_data_orig[0][40000:], train_data_orig[1][40000:]]
        model = model_struct(model_type, input_shape, 10)
        assert len(train_data[0]) == 20000 and len(val_data[0]) == 10000
        print(dataset + ' loaded.')
    elif dataset == 'cifar10-channelswitched':
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
        model = model_struct(model_type, input_shape, 10)
        assert len(train_data[0]) == 20000 and len(val_data[0]) == 10000
        print(dataset + ' loaded.')
    else:
        print('Not implemented')
        return

    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)

    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    print('*-> ' + path+weight_file)
    if not os.path.isfile(path+weight_file):
        # print('Start training')
        train_and_save(model, epochs, data_augmentation, path + weight_file, train_data, val_data, m_batch_size)

    # print('Weight file found:' + path+weight_file + ', loading.')
    model.load_weights(path + weight_file)

    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    (x_val, y_val) = format_data(val_data, 10)
    score = model.evaluate(x_val, y_val, verbose=0)
    # print('Test loss:', score[0])
    print('Val accuracy:', score[1])
    # model.summary()
    return model, model_name


def fine_tune(model, model_name, ft_train_data, ft_val_data, ft_epochs, ft_data_augmentation, nametag, path=''):

    input_shape = ft_train_data[0].shape[1:]
    # print('input shape', input_shape)
    model_type = model_name.split('_')[0]
    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)
    if ft_data_augmentation is True:
        # With DataAugmentation
        ft_model_name = model_name + '_ftwda' + str(ft_epochs) + 'ep-' + nametag
    else:
        # WithOut DataAugmentation
        ft_model_name = model_name + '_ftwoda' + str(ft_epochs) + 'ep-' + nametag
    weights_file = ft_model_name + '.h5'

    if model is None:
        model = model_struct(model_type, input_shape, 10)
        # model.load_weights(weights_file)

    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    print('*-> ' + path + weights_file)
    if not os.path.isfile(path+weights_file):
        # print('Start training')
        train_and_save(model, ft_epochs, ft_data_augmentation, path + weights_file, ft_train_data, ft_val_data,
                       m_batch_size)

    # print('Weight file found: ' + path+weights_file + ', loading.')
    model.load_weights(path + weights_file)
    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    (x_val, y_val) = format_data(ft_val_data, 10)
    score = model.evaluate(x_val, y_val, verbose=0)
    # print('Test loss:', score[0])
    print('Val accuracy:', score[1])
    # model.summary()

    return model, ft_model_name


def load_by_name(model_name, input_shape, weight_file):
    model_type = model_name.split('_')[0]
    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)
    model = model_struct(model_type, input_shape, 10)
    model.load_weights(weight_file)
    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)
    return model


def train2(model_type, tr_data, val_data, tag, epochs, data_augmentation, path=''):

    if data_augmentation is True:
        # With DataAugmentation
        model_name = '%s_%s_%dep_wda' % (model_type, tag, epochs)
    else:
        # WithOut DataAugmentation
        model_name = '%s_%s_%dep_woda' % (model_type, tag, epochs)

    print('###---> ' + model_name + ' <---###')
    weight_file = model_name + '.h5'

    input_shape = tr_data[0].shape[1:]
    model = model_struct(model_type, input_shape, 10)
    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)

    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    print('*-> ' + path+weight_file)
    if not os.path.isfile(path+weight_file):
        # print('Start training')
        train_and_save(model, epochs, data_augmentation, path + weight_file, tr_data, val_data, m_batch_size)

    # print('Weight file found:' + path+weight_file + ', loading.')
    model.load_weights(path + weight_file)

    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    (x_val, y_val) = format_data(val_data, 10)
    score = model.evaluate(x_val, y_val, verbose=0)
    # print('Test loss:', score[0])
    print('Val accuracy:', score[1])
    # model.summary()
    return model, model_name


def reg_from_(model, model_type):
    assert isinstance(model, Model)
    input = model.input
    model.layers.pop()
    model.layers.pop()
    x = GlobalAveragePooling2D()(model.layers[-1].output)
    output = Dense(1, activation="linear")(x)
    model = Model(input, output)
    # model.summary()
    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)
    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)
    return model


def train_reg(model, model_type, tr_data, val_data, tag, epochs, data_augmentation, path=''):

    if data_augmentation is True:
        # With DataAugmentation
        model_name = 'reg_%s_%s_%dep_wda' % (model_type, tag, epochs)
    else:
        # WithOut DataAugmentation
        model_name = 'reg_%s_%s_%dep_woda' % (model_type, tag, epochs)

    print('###---> ' + model_name + ' <---###')
    weight_file = model_name + '.h5'

    input_shape = tr_data[0].shape[1:]

    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)

    model.compile(loss='mean_squared_error',
                  optimizer=m_optimizer,
                  metrics=m_metric)

    print('*-> ' + path+weight_file)
    if not os.path.isfile(path+weight_file):
        # print('Start training')
        train_and_save(model, epochs, data_augmentation, path + weight_file, tr_data, val_data, m_batch_size, regression=True)

    # print('Weight file found:' + path+weight_file + ', loading.')
    model.load_weights(path + weight_file)

    model.compile(loss='mean_squared_error',
                  optimizer=m_optimizer,
                  metrics=m_metric)

    X_val, y_val = val_data
    X_val = X_val.astype('float32')
    X_val /= 255
    score = model.evaluate(X_val, y_val, verbose=0)
    # print('Test loss:', score[0])
    print('Val accuracy:', score[1])
    # model.summary()
    return model, model_name
