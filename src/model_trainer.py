import keras.applications as kapp
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
import os
import tensorflow as tf
from keras import backend as K

NUM_PARALLEL_EXEC_UNITS = 4

config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

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


def format_data(train_data, test_data, num_classes):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def train_and_save(model, epochs, data_augmentation, weight_file, train_data, test_data, batch_size):

    (x_train, y_train), (x_test, y_test) = format_data(train_data, test_data, 10)

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=2,
            shuffle=True)
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
            validation_data=(x_test, y_test),
            workers=4,
            verbose=2,
            steps_per_epoch=(50000 / batch_size)
        )

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights(weight_file)


def train(model_type, dataset, epochs, data_augmentation):

    if data_augmentation is True:
        # With DataAugmentation
        model_name = '%s_%s_%dep_wda' % (model_type, dataset, epochs)
    else:
        # WithOut DataAugmentation
        model_name = '%s_%s_%dep_woda' % (model_type, dataset, epochs)

    print('--> ' + model_name)
    weight_file = model_name + '.h5'

    if dataset == 'cifar10':
        # Load CIFAR10 data
        train_data, test_data = cifar10.load_data()
        print(dataset + ' loaded.')
        input_shape = train_data[0].shape[1:]
        print(input_shape)
        model = model_struct(model_type, input_shape, 10)
        print(model_type + ' structure loaded.')
    else:
        print 'Not implemented'
        return

    (m_batch_size, m_loss, m_optimizer, m_metric) = model_param(model_type)

    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)

    if os.path.isfile(weight_file):
        print('Weight file found, loading.')
        model.load_weights(weight_file)
    else:
        print('Start training')
        train_and_save(model, epochs, data_augmentation, weight_file, train_data, test_data, m_batch_size)

    model.compile(loss=m_loss,
                  optimizer=m_optimizer,
                  metrics=m_metric)
    #model.summary()
    return model, model_name
