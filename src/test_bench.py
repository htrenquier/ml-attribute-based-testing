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
from sklearn import preprocessing

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.chdir(os.path.dirname(sys.argv[0]))

# 'densenet169', 'densenet201',
models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50') #  , 'vgg16', 'vgg19')
# models = ('densenet121', 'mobilenetv2')
# models = ('mobilenet', 'densenet121', 'densenet169', 'densenet201')
# models = ['resnet50']
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
    """Outdated function"""
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

        aa.color_domains_accuracy(model0)

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

        cc1 = aa.color_domains_accuracy(model1)

        model2, model_name2 = mt.fine_tune(model0, model_name0, train_data_ref, val_data, 50, False, 'ref7', path=res_path)
        y_predicted = predict(model2, formatted_test_data)
        log_predictions(y_predicted, model_name2, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(formatted_test_data[1], axis=1)
        aa.accuracy(predicted_classes, true_classes)

        cc2 = aa.color_domains_accuracy(model2)

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
        f_test_data = mt.format_data(test_data, 10)  # f_~ for formatted

        model0, model_name0 = mt.train2(m, tr_data, val_data, 'cifar10-2-5', 50, data_augmentation=False, path=res_path)
        y_predicted = predict(model0, f_test_data)
        # log_predictions(y_predicted, model_name0, path=res_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        true_classes = np.argmax(f_test_data[1], axis=1)
        acc = aa.accuracy(predicted_classes, true_classes)
        print('acc', acc)

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
        # # densenet
        # ('Mean', 0.7694504688438)
        # ('Std', 0.36810717870924914)
        # ('Max', 1.0)
        # ('Min', 3.5984962468501525e-09)
        # # mobilenet
        # ('Mean', 0.7531032740719625)
        # ('Std', 0.21170846179242053)
        # ('Max', 0.997079362506271)
        # ('Min', 0.027418033376725572)
        # #
        print('Ground truth values:')
        print('Mean', np.mean(y_true))
        print('Std', np.std(y_true))
        print('Max', np.max(y_true))
        print('Min', np.min(y_true))
        y_predicted1 = model1.predict(formatted_test_data[0])
        # print(np.array(y_predicted).shape)
        n_guesses = len(y_predicted1)
        y_predicted2 = [y_predicted1[k][0] for k in xrange(n_guesses)]
        print('Prediction values:')
        print('Mean', np.mean(y_predicted2))
        print('Std', np.std(y_predicted2))
        print('Max', np.max(y_predicted2))
        print('Min', np.min(y_predicted2))
        y_predicted3 = y_predicted2 / np.linalg.norm(y_predicted2)
        print('Norm Prediction values:')
        print('Mean', np.mean(y_predicted3))
        print('Std', np.std(y_predicted3))
        print('Max', np.max(y_predicted3))
        print('Min', np.min(y_predicted3))

        fig, axs = plt.subplots(1, 1)
        axs.hist(y_true, bins=30)
        axs.set_title('y_true for ' + m)
        plt.show()

        fig, axs = plt.subplots(1, 1)
        axs.hist(y_predicted2, bins=30, range=(0, 2))
        axs.set_title(m)
        plt.show()

        diff2 = []
        diff3 = []
        for k in xrange(min(10000, len(y_predicted))):
            diff2.append(abs(y_predicted2[k] - y_true[k]))
            diff3.append(abs(y_predicted3[k] - y_true[k]))
        print('Difference:')
        print('Mean ', np.mean(diff2))
        print('Max ', max(diff2))
        print('Difference Norm:')
        print('Mean ', np.mean(diff3))
        print('Max ', max(diff3))

        # for rank, iid in enumerate(sorted_pr_args[0:10000]):
        #     q = predicted_classes[iid] == true_classes[iid]
            # if not q:
            #     print(str(iid) + ' - pr #' + str(rank) + ' =' + str(pr[iid]) + ' : ' + str(q) +
            #       '//' + str(y_predicted[iid]))

        # R/W guess prediction
        opti_thr = float(np.sort(y_predicted2)[int(acc*10000)])
        print('opti_thr', opti_thr)
        thresholds = (float(0.6), float(0.7), float(0.777), float(0.8), float(0.9), opti_thr)
        # thresholds = (float(0.9), float(1), float(1.1), float(1.2), opti_thr)

        for thr in thresholds:
            n_right_guesses = 0
            for k in xrange(n_guesses):
                q = (true_classes[20000+k] == predicted_classes[20000+k])
                p = y_predicted1[k][0] > thr
                if p == q:
                    n_right_guesses = n_right_guesses + 1

            print('acc for reg for true/false with thr of ' + str(thr) + ': ' + str(float(n_right_guesses)/n_guesses))

        # n_images = 10
        # n_rows = 10
        # for th in xrange(n_rows):
        #     fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
        #                              subplot_kw={'xticks': (), 'yticks': ()})
        #     for dec in xrange(n_images):
        #         ax = axes[dec]
        #         pr_rank = 7000 + th * 100 + dec
        #         img_id = sorted_pr_args[pr_rank]
        #         # print(str(pr_rank) + ': ' + str(y_test[img_id]))  # + ' conf. guessed = ' + str(guessed[img_id]))
        #         ax.imshow(X_test[img_id], vmin=0, vmax=1)
        #         ax.set_title('pr#' + str(pr_rank) + "\nid#" + str(img_id)
        #                      + '\nr=' + str("{0:.2f}".format(pr[img_id]))
        #                      + '\np_cl=' + str(predicted_classes[img_id])
        #                      + '\nr_cl=' + str(true_classes[img_id]))
        #     plt.show()

        print('           ~           ')


def color_region_finetuning():
    g = 4
    images_cube = ds.cifar10_maxcolor_domains(granularity=g, data_range=(50000, 60000))
    region_sizes = ds.cube_cardinals(images_cube)
    tr_data = ds.get_data('cifar10', (0, 20000))
    val_data = ds.get_data('cifar10', (40000, 50000))
    ft_data = ds.get_data('cifar10', (20000, 40000))
    train_data_ref = ds.get_data('cifar10', (20000, 30000))
    # train_data_ref2 = ds.get_data('cifar10', (30000, 40000))
    train_data_ref2 = ds.get_data('cifar10', (25000, 35000))
    test_data = ds.get_data('cifar10', (50000, 60000))
    f_test_data = mt.format_data(test_data, 10)
    ft_data_augmentation = True
    ft_epochs = 30
    
    for m in models:

        # cr = color region, 0-2 for tr data / 4-5 for val data
        model_base, model_name0 = mt.train2(m, tr_data, val_data, 'cr_0245', 50, data_augmentation=False, path=res_path)

        # model0 = model_base  # avoid fitting model_base directly DOES NOT WORK
        # model0 = mt.load_by_name(model_name0, ft_data[0].shape[1:], res_path + model_name0 + '.h5')

        # model2, model_name2 = mt.fine_tune(model0, model_name0, train_data_ref, val_data, 30, True, 'ft_2345_ref2', path=res_path)

        for x in xrange(g):
            nametag_prefix = 'ft_2345_ref' + str(x+4)
            ft_model_name = mt.fine_tune_file_name(model_name0, ft_data_augmentation, ft_epochs, nametag_prefix)
            weights_file = res_path + ft_model_name + '.h5'
            print('*-> ' + weights_file)

            if mt.model_state_exists(weights_file):
                model2 = mt.load_by_name(model_name0, ft_data[0].shape[1:], weights_file)
                (x_val, y_val) = mt.format_data(val_data, 10)
                score = model2.evaluate(x_val, y_val, verbose=0)
                # print('Test loss:', score[0])
                print('Val accuracy:', score[1])
            else:
                model0 = mt.load_by_name(model_name0, ft_data[0].shape[1:], res_path + model_name0 + '.h5')
                # #Model state check (should be same acc than base model)
                # print('Ref #' + str(x) + ' - ' + model_name0 + ' - (val_acc: '
                #       + str(model0.evaluate(x_val, y_val, verbose=0)[1]) + ')')
                assert len(train_data_ref2[0]) == 10000
                model2, model_name2 = mt.fine_tune(model0, model_name0, train_data_ref2, val_data, ft_epochs,
                                                   ft_data_augmentation, nametag_prefix, path=res_path)
            scores_cube2 = aa.color_domains_accuracy(model2, g)
            # print('Scores cube ref:', scores_cube2)
            weighted_cube = scores_cube2 * np.array(region_sizes) / float(10000)
            print('(Approx) Test accuracy', np.nansum(weighted_cube))  # Weighted average score_cube
            for y in xrange(g):
                for z in xrange(g):
                    if region_sizes[x][y][z] > 1000:
                        print('#--> Region ' + str(x)+str(y)+str(z) + ' (' + str(region_sizes[x][y][z]) + ' images)')
                        nametag_prefix = 'ft_2445_r' + str(x) + str(y) + str(z) + '_cr_1'
                        ft_model_name = mt.fine_tune_file_name(model_name0, ft_data_augmentation, ft_epochs,
                                                               nametag=nametag_prefix+'exp')
                        weights_file = res_path+ft_model_name+'.h5'
                        if mt.model_state_exists(weights_file):
                            model1 = mt.load_by_name(model_name0, ft_data[0].shape[1:], weights_file)
                            (x_val, y_val) = mt.format_data(val_data, 10)
                            score = model1.evaluate(x_val, y_val, verbose=0)
                            print('Val accuracy:', score[1])
                        else:
                            ft_data_args = aa.finetune_by_region((x, y, z), ft_data, 10000, g)
                            # Data extraction
                            dselec = np.concatenate(
                                (tr_data[0], np.array(operator.itemgetter(*ft_data_args)(ft_data[0]))))
                            dlabels = np.concatenate(
                                (tr_data[1], np.array(operator.itemgetter(*ft_data_args)(ft_data[1]))))
                            ft_data_selected = [dselec, dlabels]
                            assert len(ft_data_selected[0]) == 10000
                            # Avoid fitting model_base:
                            model0 = mt.load_by_name(model_name0, ft_data[0].shape[1:], res_path + model_name0 + '.h5')
                            # Model state check (should be same acc than base model)
                            # print('Finetuning ' + model_name0 + ' - (val_acc: '
                            #       + str(model0.evaluate(x_val, y_val, verbose=0)[1]) + ')')
                            model1, model_name1 = mt.fine_tune(model0, model_name0, ft_data_selected, val_data,
                                                               ft_epochs, ft_data_augmentation,
                                                               nametag_prefix + 'exp', path=res_path)
                        scores_cube1 = aa.color_domains_accuracy(model1, g)
                        print('  -  Region accuracy = ' + str(scores_cube1[x][y][z]))
                        weighted_cube = scores_cube1 * np.array(region_sizes) / float(10000)
                        print('  -  (Approx) Test accuracy = ', np.nansum(weighted_cube))  # Weighted average score_cube
                        cc = np.subtract(scores_cube1, scores_cube2)
                        print('  -  Region score = ' + str(cc[x][y][z]))
                        # print(cc)
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
        cube = aa.color_domains_accuracy(model0)
        print('cube', cube)
        sizes_cube = ds.cube_cardinals(cube)
        print('Sizes', sizes_cube)






check_dirs(res_path, ilsvrc2012_path)
# imagenet_test()
# finetune_test()
# data_analysis()
# bug_feature_detection()
# color_domain()
# cifar_color_domains_test()
color_region_finetuning()
