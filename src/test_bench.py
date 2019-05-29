import model_trainer as mt
import attribute_analyser as aa
import dataset_splitter as ds
import numpy as np
from keras.datasets import cifar10
import keras.applications as kapp
import tensorflow as tf
import os, sys, errno
import operator
import matplotlib.pyplot as plt
from sklearn import metrics

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.chdir(os.path.dirname(sys.argv[0]))

# 'densenet169', 'densenet201',
# models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50') #  , 'vgg16', 'vgg19')
models = ('densenet121', 'mobilenetv2')
# models = ('densenet121', 'densenet169', 'densenet201')
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
    print_size_cube_color_domain()

    for m in models:
        model0, model_name0 = mt.train(m, 'cifar10-2-5', 50, data_augmentation=False, path=res_path)
        # model0, model_name0 = mt.train(m, 'cifar10-channelswitched', 50, data_augmentation=False, path=res_path)
        y_predicted = predict(model0, formatted_test_data)
        log_predictions(y_predicted, model_name0, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        color_domains_accuracy(model0)

        pr = aa.prediction_ratings(y_predicted, true_classes)
        high_pr, low_pr = aa.sort_by_confidence(pr, len(pr) // 4)

        ft_data_src = [train_data_orig[0][training_data_len:40000], train_data_orig[1][training_data_len:40000]]
        ft_data_args = aa.finetune_by_cdc(high_pr, low_pr, test_data_orig, ft_data_src, model_name0, res_path)
        # ft_data_args = aa.finetune_by_colorfulness(ft_data_src[0], 10000, model_name0, res_path)

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

        model1, model_name1 = mt.fine_tune(model0, model_name0, ft_data_selected, val_data, 50, False, 'exp7', path=res_path)
        y_predicted = predict(model1, formatted_test_data)
        log_predictions(y_predicted, model_name1, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        cc1 = color_domains_accuracy(model1)

        model2, model_name2 = mt.fine_tune(model0, model_name0, train_data_ref, val_data, 50, False, 'ref7', path=res_path)
        y_predicted = predict(model2, formatted_test_data)
        log_predictions(y_predicted, model_name2, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        cc2 = color_domains_accuracy(model2)

        cc = np.subtract(cc1, cc2)
        print(cc)

        print('           ~           ')


def data_analysis():
    training_data_len = 20000
    train_data_orig, test_data_orig = cifar10.load_data()
    formatted_test_data = mt.format_data(test_data_orig, 10)
    # ds.print_ds_color_distrib()

    for m in models[:0]:
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
            scores.append(aa.colorfulness(image))

        max = np.max(scores)
        index = list(scores).index(max)
        print(index)
        scores.pop(index)
        pr.pop(index)

        aa.plot(pr, scores, True, res_path+model_name0+'contrast.png')

        high_pr, low_pr = aa.sort_by_confidence(pr, len(pr) // 4)


def bug_feature_detection():

    for m in models:
        tr_data = ds.get_data('cifar10', (0, 20000))
        val_data = ds.get_data('cifar10', (20000, 30000))
        test_data = ds.get_data('cifar10', (30000, 60000))
        f_test_data = mt.format_data(test_data, 10)  # f for formatted

        model0, model_name0 = mt.train2(m, tr_data, val_data, 'cifar10-2-5', 50, data_augmentation=False, path=res_path)
        y_predicted = predict(model0, f_test_data)
        # log_predictions(y_predicted, model_name0, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(f_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        print(metrics.confusion_matrix(true_classes, predicted_classes))
        pr = aa.prediction_ratings(y_predicted, true_classes)
        sorted_pr_args = np.argsort(pr)

        # print(pr[:100])
        #
        # pr_labels = np.zeros(len(y_predicted))
        # for count, id in enumerate(sorted_pr_args):
        #     pr_labels[id] = int(10*count/len(sorted_pr_args))

        # print(pr_labels[:100])

        model1 = mt.reg_from_(model0, m)
        print('Reg model created')
        X_test, y_test = test_data
        # print(np.array(pr).shape)
        tr_data = X_test[0:20000], pr[0:20000]
        val_data = X_test[20000:30000], pr[20000:30000]
        model1, model_name1 = mt.train_reg(model1, m, tr_data, val_data, '', 50, False, path=res_path)
        # score = model1.evaluate(val_data[0], val_data[1], verbose=0)
        # print('Test loss:', score[0])
        # print('Val accuracy:', score[1])
        formatted_test_data = mt.format_data(val_data, 10)
        y_true = pr[20000:30000]
        y_predicted1 = model1.predict(formatted_test_data[0])
        # print(np.array(y_predicted).shape)
        diff = []
        for k in xrange(min(100, len(y_predicted))):
            diff.append(abs(y_predicted1[k][0] - y_true[k]))
        print(np.mean(diff))
        print(max(diff))



        n_images = 10
        n_rows = 10
        for th in xrange(n_rows):
            fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
                                     subplot_kw={'xticks': (), 'yticks': ()})
            for dec in xrange(n_images):
                ax = axes[dec]
                pr_rank = th * 10 + dec
                img_id = sorted_pr_args[pr_rank]
                print(str(pr_rank) + ': ' + str(y_test[img_id]))  # + ' conf. guessed = ' + str(guessed[img_id]))
                ax.imshow(X_test[img_id], vmin=0, vmax=1)
                ax.set_title('pr#' + str(pr_rank) + "\nid#" + str(img_id)
                             + '\nr=' + str("{0:.2f}".format(pr[img_id]))
                             + '\np_cl=' + str(predicted_classes[img_id])
                             + '\nr_cl=' + str(true_classes[img_id]))
            plt.show()

        print('           ~           ')


def color_region_finetuning():
    g = 8
    images_cube = ds.cifar10_maxcolor_domains(granularity=g, data_range=(50000, 60000))
    domain_sizes = size_cube_color_domain(images_cube)

    for m in models:
        tr_data = ds.get_data('cifar10', (0, 20000))
        val_data = ds.get_data('cifar10', (40000, 50000))
        ft_data = ds.get_data('cifar10', (20000, 40000))

        model0, model_name0 = mt.train2(m, tr_data, val_data, 'cr_0245', 50, data_augmentation=False, path=res_path)

        # print('score cubes:', scores_cube)

        for x in xrange(g):
            for y in xrange(g):
                for z in xrange(g):
                    if domain_sizes[x][y][z] > 50:
                        ft_model_name = 'ft_2445_r' + str(x) + str(y) + str(z) + '_cr_1'
                        ft_data_args = aa.finetune_by_region((x, y, z), ft_data, 10000, g)

                        # data extraction
                        dselec = np.concatenate((tr_data[0], np.array(operator.itemgetter(*ft_data_args)(ft_data[0]))))
                        dlabels = np.concatenate((tr_data[1], np.array(operator.itemgetter(*ft_data_args)(ft_data[1]))))
                        ft_data_selected = [dselec, dlabels]
                        train_data_ref = ds.get_data('cifar10', (20000, 30000))

                        model1, model_name1 = mt.fine_tune(model0, model_name0, ft_data_selected, val_data, 20, False,
                                                           ft_model_name+'exp', path=res_path)
                        scores_cube1 = color_domains_accuracy(model1, g)
                        model2, model_name2 = mt.fine_tune(model0, model_name0, train_data_ref, val_data, 50, False,
                                                           ft_model_name+'ref', path=res_path)
                        scores_cube2 = color_domains_accuracy(model2, g)

                        cc = np.subtract(scores_cube1, scores_cube2)

                        print('Region=' + str(x) + str(y) + str(z) + '  -  score = ' + str(cc[x][y][z]))
                        print(cc)
                        print('           ~           ')





def color_domain_test():
    all_data_orig = ds.get_data('cifar10', (0, 20000))
    g = 4
    n_images = 5
    # images_cube = ds.cifar10_color_domains(granularity=g, frequence=0.3)
    images_cube = ds.cifar10_maxcolor_domains(granularity=g)
    images_cube_sizes = np.zeros((g, g, g))
    total = 0
    for x in xrange(g):
        for y in xrange(g):
            for z in xrange(g):
                l = len(images_cube[x][y][z])
                images_cube_sizes[x][y][z] = l
                total += l
                id_list = images_cube[x][y][z][:n_images]
                if len(id_list) > 10000:
                    print(id_list)
                    c = 0
                    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
                                             subplot_kw={'xticks': (), 'yticks': ()})
                    for img_id in id_list:
                        ax = axes[c]
                        c += 1
                        ax.imshow(all_data_orig[0][img_id], vmin=0, vmax=1)
                        ax.set_title("id#" + str(img_id))
                    plt.show()
    print(images_cube_sizes)
    print('total', total)


def color_domains_accuracy(model, granularity=4, n=1, data_range=(50000, 60000)):
    g = granularity
    images_cube = ds.cifar10_nth_maxcolor_domains(granularity=g, n=n, data_range=data_range)
    scores_cube = np.zeros((g, g, g))
    data = ds.get_data('cifar10', data_range)
    Xf, yf = mt.format_data(data, 10)
    for x in xrange(g):
        for y in xrange(g):
            for z in xrange(g):
                test_data = [[], []]
                if len(images_cube[x][y][z]) > 1:
                    for k in images_cube[x][y][z]:
                        test_data[0].append(Xf[k])
                        test_data[1].append(yf[k])
                    # print(np.array(test_data[0]).shape)
                    y_predicted = model.predict(np.array(test_data[0]))
                    predicted_classes = np.argmax(y_predicted, axis=1)
                    true_classes = np.argmax(test_data[1], axis=1)
                    acc = aa.accuracy(predicted_classes, true_classes)
                else:
                    acc = None
                scores_cube[x][y][z] = acc
    return scores_cube


def size_cube_color_domain(cube):
    """

    :param cube: cube color domain containing the lists of indexes
    :return: new cube with cardinal of indexes per area
    """
    sizes_cube = ds.cube_cardinals(cube)
    # g = granularity
    # images_cube = ds.cifar10_maxcolor_domains(granularity=g, data_range=data_range)
    # sizes_cube = np.zeros((g, g, g))
    # for x in xrange(g):
    #     for y in xrange(g):
    #         for z in xrange(g):
    #             l = len(images_cube[x][y][z])
    #             sizes_cube[x][y][z] = l
    return sizes_cube




def cifar_color_domains_test():
    for m in models:
        tr_data = ds.get_data('cifar10', (0, 20000))
        val_data = ds.get_data('cifar10', (20000, 30000))
        test_data = ds.get_data('cifar10', (30000, 60000))
        f_test_data = mt.format_data(test_data, 10)  # f for formatted

        model0, model_name0 = mt.train2(m, tr_data, val_data, 'cifar10-2-5', 50, data_augmentation=False, path=res_path)
    #
    # for m in models:
    #     model0, model_name = mt.train(m, 'cifar10', 50, data_augmentation=True)
        cube = color_domains_accuracy(model0)
        print('cube', cube)
        sizes_cube = size_cube_color_domain(cube)
        print('Sizes', sizes_cube)






check_dirs(res_path, ilsvrc2012_path)
# imagenet_test()
# finetune_test()
# data_analysis()
# bug_feature_detection()
# color_domain()
# cifar_color_domains_test()
color_region_finetuning()
