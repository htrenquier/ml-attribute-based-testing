import model_trainer as mt
import attribute_analyser as aa
import numpy as np
from keras.datasets import cifar10
import sys

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
    print('Predicitons for ' + model_file + ' written.')


def print_accuracy(y_predicted, y_test):
    predicted_classes = np.argmax(y_predicted, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    nz = np.count_nonzero(np.subtract(predicted_classes, true_classes))
    print('Accuracy = ' + str((float(len(y_test)-nz))/len(y_test)))

sys.path.extend(['~/wdir'])

models = ('densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50', 'vgg16', 'vgg19')

train_data, test_data = cifar10.load_data()
train_data, test_data = mt.format_data(train_data, test_data, 10)


for m in models:
    model0, model_name = mt.train(m, 'cifar10', 2, data_augmentation=False)
    y_predicted = predict(model0, test_data)
    log_predictions(y_predicted, model_name)
    print_accuracy(y_predicted, test_data[1])
