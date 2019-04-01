import model_trainer as mt
import attribute_analyser as aa
import numpy as np
from keras.datasets import cifar10
import tensorflow as tf
from keras import applications

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# model is compiled
# output accuracy?
def predict(model, test_data):
    print('Predicting...')
    y_predicted = model.predict(test_data[0])
    print(len(test_data[0]))
    print(len(y_predicted))
    print(len(y_predicted[20]))
    print(y_predicted[20])
    print(y_predicted[200])
    print(y_predicted[2000])
    print(y_predicted[4000])
    return y_predicted


def log_predictions(y_predicted, model_name):
    model_file = model_name + '-res.csv'
    f = open(model_file, "w+")
    for i in xrange(len(y_predicted)):
        line = '{0}, {1}, {2}, {3}\r\n'.format(str(i),
                                               str(aa.confidence(y_predicted[i])),
                                               str(np.argmax(y_predicted[i])),
                                               str(y_predicted[i]))
        f.write(line)
    f.close()
    print('Predictions for ' + model_file + ' written.')


# 'densenet169', 'densenet201',
models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50', 'vgg16', 'vgg19')

def cifar_test():
    train_data, test_data = cifar10.load_data()
    train_data, test_data = mt.format_data(train_data, test_data, 10)
    for m in models:
        model0, model_name = mt.train(m, 'cifar10', 50, data_augmentation=True)
        y_predicted = predict(model0, test_data)
        log_predictions(y_predicted, model_name)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

def imagenet_test():
    path = '/home/henri/Downloads/imagenet-val/'
    file_list = '/home/henri/Downloads/filenames.txt'
    true_classes = aa.read_ground_truth('/home/henri/Downloads/ILSVRC2012_devkit_t12/data/'
                                        'ILSVRC2012_validation_ground_truth.txt')
    model = applications.nasnet.NASNetMobile()
    predicted_classes = aa.predict_dataset(file_list, path, model, applications.nasnet.preprocess_input)
    aa.accuracy(predicted_classes, true_classes)
