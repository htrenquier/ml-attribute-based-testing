import keras.applications as kapp
from keras.preprocessing.image import ImageDataGenerator
import os
import data_tools as dt
import numpy as np

import keras.utils
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D

from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.bin import train as kr_train
from keras_retinanet.callbacks import RedirectModel
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
import cv2


class ModelConfig:
    def __init__(self, model_type, input_shape, num_classes, weights, task='classification', backbone=None):
        if task == 'classification':
            self.model_struct = model_struct(model_type, input_shape, num_classes, weights)
        elif task == 'detection':
            if model_type == 'retinanet':
                if backbone:
                    self.backbone = backbone
                else:
                    print('No backbone given')
                return


class TrainingConfig:
    def __init__(self, model_type, ):
        return




def model_param(model_type):
    """Returns compilation parameters for each model type"""
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


def model_struct(model_type, input_shape, classes, weights=None, include_top=True):
    """
    Initializes a model instance.
    :param model_type:
    :param input_shape:
    :param classes: int Number of classes
    :param weights: weights file for initialisation
    :return: instance of /model_type/
    """
    if model_type == 'densenet121':
        return kapp.densenet.DenseNet121(include_top=include_top,
                                         weights=weights,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         pooling=None,
                                         classes=classes)
    elif model_type == 'densenet169':
        return kapp.densenet.DenseNet169(include_top=include_top,
                                         weights=weights,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         pooling=None,
                                         classes=classes)
    elif model_type == 'densenet201':
        return kapp.densenet.DenseNet201(include_top=include_top,
                                         weights=weights,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         pooling=None,
                                         classes=classes)
    elif model_type == 'mobilenet':
        return kapp.mobilenet.MobileNet(include_top=include_top,
                                        weights=weights,
                                        input_tensor=None,
                                        input_shape=input_shape,
                                        pooling=None,
                                        classes=classes)
    elif model_type == 'mobilenetv2':
        return kapp.mobilenet_v2.MobileNetV2(include_top=include_top,
                                             weights=weights,
                                             input_tensor=None,
                                             input_shape=input_shape,
                                             pooling=None,
                                             classes=classes)
    elif model_type == 'nasnet':
        return kapp.nasnet.NASNetMobile(include_top=include_top,
                                        weights=weights,
                                        input_tensor=None,
                                        input_shape=input_shape,
                                        pooling=None,
                                        classes=classes)
    elif model_type == 'resnet50':
        return kapp.resnet50.ResNet50(include_top=include_top,
                                      weights=weights,
                                      input_tensor=None,
                                      input_shape=input_shape,
                                      pooling=None,
                                      classes=classes)
    elif model_type == 'vgg16':
        return kapp.vgg16.VGG16(include_top=include_top,
                                weights=weights,
                                input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif model_type == 'vgg19':
        return kapp.vgg19.VGG19(include_top=include_top,
                                weights=weights,
                                input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)


def load_imagenet_model(model_type):
    """
    Loads an ImageNet model instance
    :param model_type:
    :return: Model instance and preprocess input function
    """
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


def train_and_save(model, epochs, data_augmentation, weight_file, train_data, val_data, batch_size, regression=False):
    """
    Trains a model. Saves the best weights only cf. ModelCheckpoint callback.
    :param model: Compiled model to train
    :param epochs: int Number of epochs
    :param data_augmentation: bool for real-time data augmentation
    :param weight_file: name of the weight file
    :param train_data:
    :param val_data:
    :param batch_size:
    :param regression:
    :return: None
    """
    (x_train, y_train) = dt.format_data(train_data, 10)
    (x_val, y_val) = dt.format_data(val_data, 10)

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
            verbose=0,
            steps_per_epoch=(50000 / batch_size),
            callbacks=[checkpoint]
        )

    # score = model.evaluate(x_val, y_val, verbose=0)
    # print('Test loss:', score[0])
    # print('Val accuracy:', score[1])
    # model.save_weights(weight_file)


def weight_file_name(model_type, tag, epochs, data_augmentation, prefix='', suffix=''):
    """
    Standard for weight file name
    :param model_type:
    :param tag:
    :param epochs:
    :param data_augmentation:
    :param prefix:
    :param suffix:
    :return: Name of the weight file
    """
    name = "_".join((model_type, tag, str(epochs)+'ep', 'wda' if data_augmentation else 'woda'))
    if prefix:
        name = prefix + "_" + name
    if suffix:
        name += "_" + suffix
    print('###---> ' + name + ' <---###')
    weight_file = name + '.h5'
    return weight_file


def ft_weight_file_name(model_name, ft_data_augmentation, ft_epochs, nametag):
    """
    Builds the model weight file's name according to parameters
    :param model_name:
    :param ft_data_augmentation: bool
    :param ft_epochs: int Number of epochs
    :param nametag: extra tag for version
    :return: weight file's name
    """

    if ft_data_augmentation is True:
        # With DataAugmentation
        ft_model_name = model_name + '_ftwda' + str(ft_epochs) + 'ep-' + nametag
    else:
        # WithOut DataAugmentation
        ft_model_name = model_name + '_ftwoda' + str(ft_epochs) + 'ep-' + nametag
    return ft_model_name


def load_by_name(model_name, input_shape, weight_file_path):
    if model_state_exists(weight_file_path):
        model_type = model_name.split('_')[0]
        (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)
        model = model_struct(model_type, input_shape, 10)
        model.load_weights(weight_file_path)
        model.compile(loss=m_loss,
                      optimizer=m_optimizer,
                      metrics=m_metric)
        return model
    else:
        raise IOError('File ' + weight_file_path + ' not found')


def model_state_exists(weight_file_path):
    """
    Check for model version weights based on file name
    :param weight_file_path: Name of the file
    :return: bool True if the file exists
    """
    return os.path.isfile(weight_file_path)


def train2(model_type, tr_data, val_data, epochs, data_augmentation, tag='', path='', weights_file=None):
    """
    Instantiates and trains a model. First checks is it exists.
    If weights is set, it loads the pre-trained state of the model (for fine tuning).
    :param model_type:
    :param tr_data: training data
    :param val_data: validation data
    :param epochs: number of training epochs
    :param data_augmentation: bool for data_augmentation
    :param tag: additional tag for the weight file's name
    :param path: path for storing result weight file
    :param weights_file: weights of previous model's state (for additional training)
    :return: trained model instance and its weight file name without extension
    """
    input_shape = tr_data[0].shape[1:]

    if weights_file:
        new_weights_file = weights_file.rstrip('.h5') + ('_ftwda' if data_augmentation else '_ftwoda') \
                      + str(epochs) + 'ep-' + tag + '.h5'
    else:
        new_weights_file = weight_file_name(model_type, tag, epochs, data_augmentation)

    model = model_struct(model_type, input_shape, 10)
    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)
    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    print('*-> ' + path + new_weights_file)
    if model_state_exists(path + new_weights_file):
        model.load_weights(path + new_weights_file)
    else:
        if weights_file:
            model.load_weights(path + weights_file)
        train_and_save(model, epochs, data_augmentation, path + new_weights_file, tr_data, val_data, m_batch_size)

    model.load_weights(path + new_weights_file)  # Loading best state according to val_acc

    # (x_val, y_val) = dt.format_data(val_data, 10)
    # score = model.evaluate(x_val, y_val, verbose=0)
    # print('Val loss:', score[0])
    # print('Val acc:', score[1])
    # model.summary()
    return model, new_weights_file.rstrip('.h5')


def reg_from_(model, model_type):
    """
    Builds a regression model from a classification model. Keeps the classification weights.
    :param model: Classification model
    :param model_type:
    :return: a regression model pretrained with classification data.
    """
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

    weight_file = weight_file_name(model_type, tag, epochs, data_augmentation, 'reg_')

    input_shape = tr_data[0].shape[1:]

    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)

    model.compile(loss='mean_squared_error',
                  optimizer=m_optimizer,
                  metrics=m_metric)

    print('*-> ' + path+weight_file)
    if not os.path.isfile(path+weight_file):
        # print('Start training')
        train_and_save(model, epochs, data_augmentation, path + weight_file, tr_data, val_data, m_batch_size,
                       regression=True)

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
    return model, weight_file.strip('.h5')


def create_generators(train_annotations, val_annotations, class_mapping, preprocess_image, batch_size,
                      data_augmentation=False, base_dir=None):
    if data_augmentation:
        transform_generator = kr_train.random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = kr_train.random_transform_generator(flip_x_chance=0.5)

    # create the generators
    train_generator = CSVGenerator(
        train_annotations,
        class_mapping,
        transform_generator=transform_generator,
        base_dir=base_dir,
        preprocess_image=preprocess_image,
        batch_size=batch_size
    )

    if val_annotations:
        validation_generator = CSVGenerator(
            val_annotations,
            class_mapping,
            base_dir=base_dir,
            preprocess_image=preprocess_image,
            batch_size=batch_size
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def create_callbacks(model, batch_size, weight_file=None, tensorboard_dir=None, snapshots_path=None,
                     backbone=None, dataset_type=None):
    callbacks = []
    if tensorboard_dir:
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=False,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    # save the model
    if snapshots_path:
        # ensure directory created first; otherwise h5py will error after epoch.
        checkpoint = ModelCheckpoint(
            os.path.join(
                snapshots_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=backbone,
                                                                    dataset_type=dataset_type)
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
    else:
        if not weight_file:
            weight_file = 'retinanet_unnamed.h5'
        checkpoint = ModelCheckpoint(
            weight_file,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto'
        )

    callbacks.append(checkpoint)

    callbacks.append(ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras'
    """
    def __init__(self, list_ids, labels, batch_size=32, dim=(64,64,3), n_classes=10, shuffle=True):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_ids = list_ids
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, id in enumerate(list_ids_temp):
            # Store sample
            # X[i, ] = np.load('data/' + id + '.npy')
            X[i, ] = cv2.imread(id)
            # Store class
            y[i] = self.labels[id]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
