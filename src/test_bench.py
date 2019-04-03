import model_trainer as mt
import attribute_analyser as aa
import numpy as np
from keras.datasets import cifar10
import tensorflow as tf
from keras import applications
import os, sys, errno


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.chdir(os.path.dirname(sys.argv[0]))

# 'densenet169', 'densenet201',
models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50', 'vgg16', 'vgg19')
ilsvrc2012_val_path = '/home/henri/Downloads/imagenet-val/'
ilsvrc2012_val_labels = '../ilsvrc2012/val_ground_truth.txt'
ilsvrc2012_path = '../ilsvrc2012'
res_path = '../res'


def check_dirs(*paths):
    for p in paths:
        try:
            os.mkdir(p)
        except OSError, e:
            if e.errno == errno.EEXIST:
                print ("Directory %s exists" % p)
            else:
                print ("Creation of the directory %s failed" % p)
        else:
            print ("Successfully created the directory %s " % p)


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


def log_predictions(y_predicted, model_name, path):
    model_file = model_name + '-res.csv'
    f = open(path+model_file, "w+")
    for i in xrange(len(y_predicted)):
        line = '{0}, {1}, {2}, {3}\r\n'.format(str(i),
                                               str(aa.confidence(y_predicted[i])),
                                               str(np.argmax(y_predicted[i])),
                                               str(y_predicted[i]))
        f.write(line)
    f.close()
    print('Predictions for ' + model_file + ' written.')


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


# https://gist.githubusercontent.com/maraoz/388eddec39d60c6d52d4/raw/791d5b370e4e31a4e9058d49005be4888ca98472/gistfile1.txt
# index to label
def imagenet_test():
    file_names, true_classes = aa.read_ground_truth(ilsvrc2012_val_labels)
    model = applications.nasnet.NASNetMobile()
    y_predicted = aa.predict_dataset(file_names, ilsvrc2012_val_path, model, applications.nasnet.preprocess_input)
    log_predictions(y_predicted, model_name='nasnet_imagenet')
    predicted_classes = np.argmax(y_predicted, axis=1)
    aa.accuracy(predicted_classes, true_classes)


check_dirs(res_path, ilsvrc2012_path)
#imagenet_test()
