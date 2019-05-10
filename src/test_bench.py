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
models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50') #  , 'vgg16', 'vgg19')
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
    return y_predicted


def log_predictions(y_predicted, model_name, path):
    model_file = model_name + '-res.csv'
    if not os.path.isfile(path + model_file):
        f = open(path + model_file, "w+")
        for i in xrange(len(y_predicted)):
            line = '{0}, {1}, {2}, {3}\r\n'.format(str(i),
                                                   str(aa.confidence(y_predicted[i])),
                                                   str(np.argmax(y_predicted[i])),
                                                   str(y_predicted[i]))
            f.write(line)
        f.close()
        # print('Predictions for ' + model_file + ' written.')
    # else:
        # print('Predictions for ' + model_file + ' already written!')



def cifar_test():
    train_data, test_data = cifar10.load_data()
    test_data = mt.format_data(test_data, 10)
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
    training_data_len = 20000
    train_data_orig, test_data_orig = cifar10.load_data()

    # train_img_switch = []
    # test_img_switch = []
    # for img in train_data_orig[0]:
    #     train_img_switch.append(np.roll(img, 1, 2))
    # for img in test_data_orig[0]:
    #     test_img_switch.append(np.roll(img, 1, 2))
    # train_data_orig[0][:] = np.array(train_img_switch)
    # test_data_orig[0][:] = np.array(test_img_switch)

    formatted_test_data = mt.format_data(test_data_orig, 10)

    for m in models:
        model0, model_name0 = mt.train(m, 'cifar10-2-5', 50, data_augmentation=False, path=res_path)
        # model0, model_name0 = mt.train(m, 'cifar10-channelswitched', 50, data_augmentation=False, path=res_path)
        y_predicted = predict(model0, formatted_test_data)
        log_predictions(y_predicted, model_name0, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        pr = aa.prediction_ratings(y_predicted, true_classes)
        high_pr, low_pr = aa.sort_by_confidence(pr, len(pr) // 4)

        ft_data_src = [train_data_orig[0][training_data_len:40000], train_data_orig[1][training_data_len:40000]]
        # ft_data_args = aa.finetune_by_cdc(high_pr, low_pr, test_data_orig, ft_data_src, model_name0, res_path)
        ft_data_args = aa.finetune_by_colorfulness(ft_data_src[0], 10000, model_name0, res_path)

        print(ft_data_args)

        # print(finetune_data_args)
        dselec = np.concatenate((train_data_orig[0][:training_data_len],
                              np.array(operator.itemgetter(*ft_data_args)(ft_data_src[0]))))
        dlabels = np.concatenate((train_data_orig[1][:training_data_len],
                              np.array(operator.itemgetter(*ft_data_args)(ft_data_src[1]))))

        ft_data_selected = [dselec, dlabels]

        train_data_ref = [train_data_orig[0][:training_data_len+10000],
                          train_data_orig[1][:training_data_len+10000]]

        val_data = [train_data_orig[0][-10000:], train_data_orig[1][-10000:]]

        assert len(ft_data_selected) == 2 and len(ft_data_selected[0]) == 30000

        model1, model_name1 = mt.fine_tune(model0, model_name0, ft_data_selected, val_data, 50, False, 'exp-clfn', path=res_path)
        y_predicted = predict(model1, formatted_test_data)
        log_predictions(y_predicted, model_name1, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        model2, model_name2 = mt.fine_tune(model0, model_name0, train_data_ref, val_data, 50, False, 'ref-clfn', path=res_path)
        y_predicted = predict(model2, formatted_test_data)
        log_predictions(y_predicted, model_name2, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)


def data_analysis():
    training_data_len = 20000
    train_data_orig, test_data_orig = cifar10.load_data()
    formatted_test_data = mt.format_data(test_data_orig, 10)

    for m in models:
        model0, model_name0 = mt.train(m, 'cifar10-2-5', 50, data_augmentation=False, path=res_path)
        # model0, model_name0 = mt.train(m, 'cifar10-channelswitched', 50, data_augmentation=False, path=res_path)
        y_predicted = predict(model0, formatted_test_data)
        log_predictions(y_predicted, model_name0, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        pr = aa.prediction_ratings(y_predicted, true_classes)
        scores = []

        for image in test_data_orig[0]:
            scores.append(aa.contrast(image))

        max = np.max(scores)
        index = np.where(scores == max)
        del(scores[index])
        del(pr[index])
        
        aa.plot(pr, scores, True, res_path+model_name0+'contrast.png')

        high_pr, low_pr = aa.sort_by_confidence(pr, len(pr) // 4)




check_dirs(res_path, ilsvrc2012_path)
# imagenet_test()
# finetune_test()
data_analysis()