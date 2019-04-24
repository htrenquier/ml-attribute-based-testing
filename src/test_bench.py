import model_trainer as mt
import attribute_analyser as aa
import numpy as np
from keras.datasets import cifar10
import keras.applications as kapp
import tensorflow as tf
import os, sys, errno
import operator

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.chdir(os.path.dirname(sys.argv[0]))

# 'densenet169', 'densenet201',
models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50', 'vgg16', 'vgg19')
ilsvrc2012_val_path = '/home/henri/Downloads/imagenet-val/'
ilsvrc2012_val_labels = '../ilsvrc2012/val_ground_truth.txt'
ilsvrc2012_path = '../ilsvrc2012/'
res_path = '../res/'


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
        log_predictions(y_predicted, model_name, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)


# https://gist.githubusercontent.com/maraoz/388eddec39d60c6d52d4/raw/791d5b370e4e31a4e9058d49005be4888ca98472/gistfile1.txt
# index to label
def imagenet_test():
    file_names, true_classes = aa.read_ground_truth(ilsvrc2012_val_labels)
    for m in models:
        model, preprocess_func = mt.load_imagenet_model(m)
        y_predicted = aa.predict_dataset(file_names, ilsvrc2012_val_path, model, preprocess_func)
        log_predictions(y_predicted, model_name=m+'_imagenet', path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        aa.accuracy(predicted_classes, true_classes)


def finetune_test():
    training_data_len = 30000
    train_data, test_data = cifar10.load_data()
    train_data, test_data = mt.format_data(train_data, test_data, 10)
    train_data_ = train_data[:training_data_len]

    for m in models:
        model0, model_name = mt.train(m, 'cifar10-3/5', 50, data_augmentation=False)
        y_predicted = predict(model0, test_data)
        log_predictions(y_predicted, model_name, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        pr = aa.prediction_ratings(y_predicted, true_classes)
        high_pr, low_pr = aa.sort_by_confidence(pr, len(pr) // 4)

        print('len training data:', len(train_data_))
        cdc_high = aa.ColorDensityCube(resolution=8)
        for img in aa.get_images(high_pr, train_data_):
            cdc_high.feed(img)
        # cdc_train.avg()
        cdc_high.normalize()
        cdc_high.plot_cube(save=True, title=model_name + 'high_pr')

        cdc_low = aa.ColorDensityCube(resolution=8)
        for img in aa.get_images(low_pr, train_data_):
            cdc_low.feed(img)
        # cdc_train.avg()
        cdc_low.normalize()
        cdc_low.plot_cube(save=True, title=model_name + 'low_pr')

        cdc_diff = cdc_high.substract(cdc_low, state='norm')  # What does high has more than low?
        # cdc_diff.plot_cube()

        cdc_finetune = aa.ColorDensityCube(resolution=8)
        ft_data = train_data[training_data_len:]
        finetune_data_args = aa.get_best_scores(ft_data, 10000, cdc_diff)
        for img_index in finetune_data_args:
            cdc_finetune.feed(train_data_[img_index])
        cdc_finetune.normalize()
        cdc_finetune.plot_cube(save=True, title=model_name + 'ft_selection')

        ft_data_selected = operator.itemgetter(*finetune_data_args)(ft_data)

        model1, model_name1 = mt.fine_tune(m, ft_data_selected, 20, False, 'exp')
        y_predicted = predict(model1, test_data)
        log_predictions(y_predicted, model_name1, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        model2, model_name2 = mt.fine_tune(m, ft_data, 20, False, 'ref')
        y_predicted = predict(model2, test_data)
        log_predictions(y_predicted, model_name2, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)



check_dirs(res_path, ilsvrc2012_path)
#imagenet_test()
finetune_test()
