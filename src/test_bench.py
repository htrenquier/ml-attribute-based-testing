from __future__ import division
import numpy as np
from keras.datasets import cifar10
import tensorflow as tf
import os, sys, errno
import matplotlib.pyplot as plt
import operator
from sklearn import metrics as sk_metrics
import metrics
import metrics_color
import plotting
import model_trainer as mt
import data_tools as dt
import tests_logging as t_log
import bdd100k_utils as bu

# import keras_retinanet
from keras_retinanet import models as kr_models
from keras_retinanet.bin import train as kr_train

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.chdir(os.path.dirname(sys.argv[0]))

# 'densenet169', 'densenet201',
# models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50')
# models = ('densenet121', 'mobilenetv2')
# models = ('mobilenet', 'densenet121', 'densenet169', 'densenet201')
# models = ['densenet121']
models = ['resnet50']  # ,'mobilenet128_1']
ilsvrc2012_val_path = '/home/henri/Downloads/imagenet-val/'
ilsvrc2012_val_labels = '../ilsvrc2012/val_ground_truth.txt'
ilsvrc2012_path = '../ilsvrc2012/'
res_path = '../res/'
h5_path = '../res/h5/'
csv_path = '../res/csv/'
png_path = '../res/png/'
tb_path = '../res/logs/'
bdd100k_labels_path = "../../bdd100k/labels/"
bdd100k_data_path = "../../bdd100k/images/100k/"
bdd100k_val_path = "../../bdd100k/images/100k/val/"
bdd100k_train_path = "../../bdd100k/images/100k/train/"


def check_dirs(*paths):
    print(os.getcwd())
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
# def predict(model, test_data):
#     print('Predicting...')
#     y_predicted = model.predict(test_data[0])
#     return y_predicted


def cifar_test():
    train_data, test_data = cifar10.load_data()
    for m in models:
        model0, model_name = mt.train2(m, train_data, test_data, 50, True, 'cifar10', h5_path)
        # model0, model_name = mt.train(m, 'cifar10', 50, data_augmentation=True)
        # y_predicted = predict(model0, test_data)
        acc, _, y_predicted = metrics.predict_and_acc(model0, test_data)
        t_log.log_predictions(y_predicted, model_name, file_path=csv_path)
        # predicted_classes = np.argmax(y_predicted, axis=1)
        # true_classes = np.argmax(test_data[1], axis=1)
        # metrics.accuracy(predicted_classes, true_classes)


# https://gist.githubusercontent.com/maraoz/388eddec39d60c6d52d4/raw/791d5b370e4e31a4e9058d49005be4888ca98472/gistfile1.txt
# index to label
def imagenet_test():
    file_names, true_classes = t_log.read_ground_truth(ilsvrc2012_val_labels)
    for m in models:
        model, preprocess_func = mt.load_imagenet_model(m)
        y_predicted = dt.predict_dataset(file_names, ilsvrc2012_val_path, model, preprocess_func)
        t_log.log_predictions(y_predicted, model_name=m + '_imagenet', file_path=csv_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        metrics.accuracy(predicted_classes, true_classes)


# def finetune_test():
#     """Outdated function"""
#     training_data_len = 20000
#     train_data_orig, test_data_orig = cifar10.load_data()
#
#     # train_img_switch = []
#     # test_img_switch = []
#     # for img in train_data_orig[0]:
#     #     train_img_switch.append(np.roll(img, 1, 2))
#     # for img in test_data_orig[0]:
#     #     test_img_switch.append(np.roll(img, 1, 2))
#     # train_data_orig[0][:] = np.array(train_img_switch)
#     # test_data_orig[0][:] = np.array(test_img_switch)
#
#     formatted_test_data = mt.format_data(test_data_orig, 10)
#
#     for m in models:
#         model0, model_name0 = mt.train(m, 'cifar10-2-5', 50, data_augmentation=False, path=res_path)
#         # model0, model_name0 = mt.train(m, 'cifar10-channelswitched', 50, data_augmentation=False, path=res_path)
#         y_predicted = predict(model0, formatted_test_data)
#         logg.log_predictions(y_predicted, model_name0, path=res_path)
#         predicted_classes = np.argmax(y_predicted, axis=1)
#         true_classes = np.argmax(formatted_test_data[1], axis=1)
#         metrics.accuracy(predicted_classes, true_classes)
#
#         metrics_color.color_domains_accuracy(model0)
#
#         pr = metrics.prediction_ratings(y_predicted, true_classes)
#         high_pr, low_pr = metrics.sort_by_confidence(pr, len(pr) // 4)
#
#         ft_data_src = [train_data_orig[0][training_data_len:40000], train_data_orig[1][training_data_len:40000]]
#         ft_data_args = metrics_color.finetune_by_cdc(high_pr, low_pr, test_data_orig, ft_data_src, model_name0, res_path)
#         # ft_data_args = aa.finetune_by_colorfulness(ft_data_src[0], 10000, model_name0, res_path)
#
#         print(ft_data_args)
#
#         # print(finetune_data_args)
#         dselec = np.concatenate((train_data_orig[0][:training_data_len],
#                               np.array(operator.itemgetter(*ft_data_args)(ft_data_src[0]))))
#         dlabels = np.concatenate((train_data_orig[1][:training_data_len],
#                               np.array(operator.itemgetter(*ft_data_args)(ft_data_src[1]))))
#
#         ft_data_selected = [dselec, dlabels]
#
#         train_data_ref = [train_data_orig[0][:training_data_len+10000],
#                           train_data_orig[1][:training_data_len+10000]]
#
#         val_data = [train_data_orig[0][-10000:], train_data_orig[1][-10000:]]
#
#         assert len(ft_data_selected) == 2 and len(ft_data_selected[0]) == 30000
#
#         model1, model_name1 = mt.fine_tune(model0, model_name0, ft_data_selected, val_data, 50, False, 'exp7', path=res_path)
#         y_predicted = predict(model1, formatted_test_data)
#         logg.log_predictions(y_predicted, model_name1, file_path=res_path)
#         predicted_classes = np.argmax(y_predicted, axis=1)
#         true_classes = np.argmax(formatted_test_data[1], axis=1)
#         metrics.accuracy(predicted_classes, true_classes)
#
#         cc1 = metrics_color.color_domains_accuracy(model1)
#
#         model2, model_name2 = mt.fine_tune(model0, model_name0, train_data_ref, val_data, 50, False, 'ref7', path=res_path)
#         y_predicted = predict(model2, formatted_test_data)
#         logg.log_predictions(y_predicted, model_name2, file_path=res_path)
#         predicted_classes = np.argmax(y_predicted, axis=1)
#         true_classes = np.argmax(formatted_test_data[1], axis=1)
#         metrics.accuracy(predicted_classes, true_classes)
#
#         cc2 = metrics_color.color_domains_accuracy(model2)
#
#         cc = np.subtract(cc1, cc2)
#         print(cc)
#
#         print('           ~           ')


def data_analysis():

    tr_data = dt.get_data('cifar10', (0, 20000))
    val_data = dt.get_data('cifar10', (40000, 50000))
    test_data = dt.get_data('cifar10', (50000, 60000))

    for m in models:
        # model0, model_name0 = mt.train2(m, tr_data, val_data, 50, False, 'cifar10-2-5', h5_path)
        # model0, model_name0 = mt.train(m, 'cifar10-channelswitched', 50, data_augmentation=False, path=res_path)
        # acc, predicted_classes, y_predicted = dt.predict_and_acc(model0, test_data)
        # t_log.log_predictions(y_predicted, model_name0, file_path=csv_path)

        model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
        y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)

        # true_classes = np.argmax(test_data[1], axis=1)  # wrong
        true_classes = [int(k) for k in test_data[1]]
        pr = metrics.prediction_ratings(y_predicted, true_classes)
        scores = []

        for image in test_data[0]:
            scores.append(metrics_color.colorfulness(image))

        max = np.max(scores)
        index = list(scores).index(max)
        scores.pop(index)
        pr.pop(index)

        plotting.quick_plot(pr, scores, png_path+model_name0+'contrast.png')


def pr_on_fair_distribution(models=['densenet121'], top_n=100, res=4):
    test_data = dt.get_data('cifar10', (50000, 60000))

    # Add every image's cube in densities
    densities = []
    for img in test_data[0]:
        cc = metrics_color.ColorDensityCube(res)
        cc.feed(img)
        densities.append(cc.get_cube())
        # ccf = np.array(cc.get_cube()).flatten()

    # Shape densities (list of cubes) to make a list per color
    densities_lists = np.swapaxes(np.swapaxes(np.swapaxes(densities, 0, 3), 0, 2), 0, 1)
    # print(densities_lists.shape)
    densities_cube = np.empty((res, res, res), dtype=object)

    # For each color keep the ids of the top_n most dense images in this color (same image can be in 2 colors)
    for i in xrange(res):
        for j in xrange(res):
            for k in xrange(res):
                # pr_most_dense = []
                density_list = densities_lists[i][j][k].tolist()
                args_most_dense = np.argsort(density_list)[-top_n:]
                densities_cube[i][j][k] = args_most_dense
    # print(densities_cube.shape)

    # Per model analysis
    for m in models:
        # Load model predictions and ground_truth values
        model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
        y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
        true_classes = [int(k) for k in test_data[1]]
        pr = metrics.prediction_ratings(y_predicted, true_classes)

        # For each color get prediction score of the top_n images
        score_cube = np.zeros((res, res, res))
        global_cc = metrics_color.ColorDensityCube(resolution=res)
        args_most_dense_all = []
        for i in xrange(res):
            for j in xrange(res):
                for k in xrange(res):
                    pr_most_dense = []
                    densities_args = densities_cube[i][j][k].tolist()
                    # args_most_dense = np.argsort(density_list)[-topn:]
                    ijk_cc = metrics_color.ColorDensityCube(res)
                    for a in densities_cube[i][j][k].tolist():
                        pr_most_dense.append(pr[a])
                        ijk_cc.feed(test_data[0][a])
                        global_cc.feed(test_data[0][a])
                    ijk_cc.normalize()
                    ttl = 'color = (' + str(float(i/res)) + ', ' + str(float(j/res)) + ', ' + str(float(k/res)) + ')'
                    # ijk_cc.plot_cube()
                    score_cube[i][j][k] = np.mean(pr_most_dense)
                    print(np.mean(pr_most_dense))
                    # args_most_dense_all.append(args_most_dense)
                    ttl = 'color = (' + str(float(i/res)) + ', ' + str(float(j/res)) + ', ' + str(float(k/res)) + ')'
                    # plotting.show_imgs(densities_args[:10], ttl, test_data[0], showColorCube=True, resolution=4)

        global_cc.normalize()
        global_cc.plot_cube(title='Fair distributed dataset ColorCube')


        sc = metrics_color.ColorDensityCube(resolution=res, cube=score_cube)
        sc.normalize()
        sc.plot_cube(title='Scores per color for ' + m)


def bug_feature_detection():

    for m in models:
        tr_data = dt.get_data('cifar10', (0, 20000))
        val_data = dt.get_data('cifar10', (20000, 30000))
        test_data = dt.get_data('cifar10', (30000, 60000))

        model0, model_name0 = mt.train2(m, tr_data, val_data, 50, False, tag='cifar10-2-5', path=h5_path)
        acc, predicted_classes, y_predicted = dt.predict_and_acc(model0, test_data)
        # log_predictions(y_predicted, model_name0, path=csv_path)
        print('acc', acc)

        # print(sk_metrics.confusion_matrix(test_data[1], predicted_classes))
        # true_classes = np.argmax(test_data[1], axis=1) wrong
        true_classes = [int(k) for k in test_data[1]]
        pr = metrics.prediction_ratings(y_predicted, true_classes)

        model2, model_name2 = mt.train2(m, tr_data, val_data, 1, False, tag='cifar10-0223', path=h5_path)
        model1 = mt.reg_from_(model2, m)
        print('Reg model created')
        X_test, y_test = test_data
        tr_data = X_test[0:20000], pr[0:20000]
        val_data = X_test[20000:30000], pr[20000:30000]
        model1, model_name1 = mt.train_reg(model1, m, tr_data, val_data, '', 50, False, path=h5_path)
        # score = model1.evaluate(val_data[0], val_data[1], verbose=0)
        # print('Test loss:', score[0])
        # print('Val accuracy:', score[1])
        formatted_test_data = dt.format_data(val_data, 10)
        y_true = pr[20000:30000]
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

        # fig, axs = plt.subplots(1, 1)
        # axs.hist(y_true, bins=30)
        # axs.set_title('y_true for ' + m)
        # plt.show()
        #
        # fig, axs = plt.subplots(1, 1)
        # axs.hist(y_predicted2, bins=30, range=(0, 2))
        # axs.set_title(m)
        # plt.show()

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

        # R/W guess prediction
        opti_thr = float(np.sort(y_predicted2)[int(acc*10000)])
        print('opti_thr', opti_thr)
        thresholds = (float(0.6), float(0.7), float(0.777), float(0.8), float(0.9), opti_thr)
        # thresholds = (float(0.9), float(1), float(1.1), float(1.2), opti_thr)

        for thr in thresholds:
            n_right_guesses = 0
            for k in xrange(n_guesses):
                q = (test_data[1][20000+k] == predicted_classes[20000+k])
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
    images_cube = dt.cifar10_maxcolor_domains(granularity=g, data_range=(50000, 60000))
    region_sizes = dt.cube_cardinals(images_cube)
    tr_data = dt.get_data('cifar10', (0, 20000))
    val_data = dt.get_data('cifar10', (40000, 50000))
    ft_data = dt.get_data('cifar10', (20000, 40000))
    train_data_ref = dt.get_data('cifar10', (20000, 30000))
    train_data_ref2 = dt.get_data('cifar10', (30000, 40000))
    # train_data_ref2 = ds.get_data('cifar10', (25000, 35000))
    test_data = dt.get_data('cifar10', (50000, 60000))
    f_test_data = dt.format_data(test_data, 10)
    ft_data_augmentation = True
    ft_epochs = 30
    
    for m in models:

        # cr = color region, 0-2 for tr data / 4-5 for val data
        model_base, model_name0 = mt.train2(m, tr_data, val_data,  50, False, 'cr_0245', path=h5_path)
        scores_cubes = []

        for x in xrange(g):
            nametag_prefix = 'ft_2345_ref' + str(x+4)
            ft_model_name = mt.ft_weight_file_name(model_name0, ft_data_augmentation, ft_epochs, nametag_prefix)
            weights_file = h5_path + ft_model_name + '.h5'
            print('*-> ' + weights_file)

            if mt.model_state_exists(weights_file):
                model2 = mt.load_by_name(model_name0, ft_data[0].shape[1:], weights_file)
                score = dt.predict_and_acc(model2, val_data)
                print('Val accuracy:', score[0])
            else:
                ft_data_selected_ref = [np.concatenate((tr_data[0], train_data_ref2[0])),
                                        np.concatenate((tr_data[1], train_data_ref2[1]))]
                assert len(ft_data_selected_ref[0]) == 30000
                model2, model_name2 = mt.train2(m, ft_data_selected_ref, val_data, ft_epochs, ft_data_augmentation,
                                                nametag_prefix, h5_path, weights_file=model_name0 + '.h5')
            scores_cube2 = metrics_color.color_domains_accuracy(model2, g)
            # print('Scores cube ref:', scores_cube2)
            weighted_cube = scores_cube2 * np.array(region_sizes) / float(10000)
            print('(Approx) Test accuracy', np.nansum(weighted_cube))  # Weighted average score_cube
            scores_cubes.append(scores_cube2)

        avg_ref_score_cube = np.nanmean(scores_cubes, axis=0)
        max_ref_score_cube = np.max(scores_cubes, axis=0)

        for x in xrange(g):
            for y in xrange(g):
                for z in xrange(g):
                    if region_sizes[x][y][z] > 100:
                        print('#--> Region ' + str(x)+str(y)+str(z) + ' (' + str(region_sizes[x][y][z]) + ' images)')
                        nametag_prefix = 'ft_2445_r' + str(x) + str(y) + str(z) + '_cr_1'

                        ft_model_name = mt.ft_weight_file_name(model_name0, ft_data_augmentation, ft_epochs,
                                                               nametag=nametag_prefix+'exp')
                        weights_file = h5_path+ft_model_name+'.h5'

                        if mt.model_state_exists(weights_file):
                            model1 = mt.load_by_name(model_name0, ft_data[0].shape[1:], weights_file)
                            score = dt.predict_and_acc(model1, val_data)
                            print('Val accuracy:', score[0])
                        else:
                            ft_data_args = metrics_color.finetune_by_region((x, y, z), ft_data, 10000, g)
                            ft_data_selected = dt.get_finetune_data(tr_data, ft_data, ft_data_args)
                            assert len(ft_data_selected[0]) == 30000
                            model1, model_name1 = mt.train2(m, ft_data_selected, val_data, ft_epochs,
                                                            ft_data_augmentation, nametag_prefix + 'exp',
                                                            h5_path, weights_file=model_name0 + '.h5')
                        scores_cube1 = metrics_color.color_domains_accuracy(model1, g)
                        # print('Scores cube exp:', scores_cube1)
                        print('  -  Region accuracy = ' + str(scores_cube1[x][y][z]))
                        weighted_cube = scores_cube1 * np.array(region_sizes) / float(10000)
                        print('  -  (Approx) Test accuracy = ', np.nansum(weighted_cube))  # Weighted average score_cube
                        # cc = np.subtract(scores_cube1, scores_cube2)
                        cc_avg = np.subtract(scores_cube1, avg_ref_score_cube)
                        print('  -  Region score (avg ref) = ' + str(float(cc_avg[x][y][z])))
                        cc_max = np.subtract(scores_cube1, max_ref_score_cube)
                        print('  -  Region score (max ref) = ' + str(float(cc_max[x][y][z])))
                        # print(cc)
                        print('           ~           ')


def retinanet_test():
    labels_to_names = {0: 'bus', 1: 'traffic light', 2: 'traffic sign', 3: 'person', 4: 'bike', 5: 'truck', 6: 'motor',
                       7: 'car', 8: 'train', 9: 'rider'}

    model.fit_generator()



def color_domain_test():
    all_data_orig = dt.get_data('cifar10', (0, 20000))
    g = 4
    n_images = 5
    # images_cube = ds.cifar10_color_domains(granularity=g, frequence=0.3)
    images_cube = dt.cifar10_maxcolor_domains(granularity=g)
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
        tr_data = dt.get_data('cifar10', (0, 20000))
        val_data = dt.get_data('cifar10', (20000, 30000))
        test_data = dt.get_data('cifar10', (30000, 60000))
        f_test_data = dt.format_data(test_data, 10)  # f for formatted

        model0, model_name0 = mt.train2(m, tr_data, val_data, 50, False, 'cifar10-2-5', path=h5_path)
    #
    # for m in models:
    #     model0, model_name = mt.train(m, 'cifar10', 50, data_augmentation=True)
        cube = metrics_color.color_domains_accuracy(model0)
        print('cube', cube)
        sizes_cube = dt.cube_cardinals(cube)
        print('Sizes', sizes_cube)


def mt_noise_test():
    np.random.seed(0)
    tr_data = dt.get_data('cifar10', (0, 40000))
    val_data = dt.get_data('cifar10', (40000, 50000))
    for noise_level in xrange(5, 200, 10):
        for k in [1]:  #xrange(len(tr_data[0])):
            # noise_mat = np.repeat(np.random.random((32, 32))[:, :, np.newaxis], 3, axis=2)
            noise_mat = np.swapaxes([np.random.random((32, 32)), np.random.random((32, 32)), np.random.random((32, 32))]
                                    , 0, 2)
            print(tr_data[0][k].shape)
            print(noise_mat.shape)
            tr_data[0][k] = np.clip(tr_data[0][k].astype('uint16') * (1 + (noise_mat-0.5) * noise_level/100), 0, 255)\
                .astype('uint8')
            plotting.imshow(tr_data[0][k])
        for m in models:
            print('Training', m)
            # model0, model_name0 = mt.train2(m, tr_data, val_data, 40, False,
            #                                 'cifar_mt_0445_noise2_' + str(noise_level), path=h5_path)
            # acc, _, _ = dt.predict_and_acc(model0, val_data)
            # print('Validation accuracy = ', acc)
            # print(model_name0, 'trained')


def epochs_accuracy_test():
    tr_data = dt.get_data('cifar10', (0, 40000))
    val_data = dt.get_data('cifar10', (40000, 50000))
    test_data = dt.get_data('cifar10', (50000, 60000))
    m = models[0]
    epochs = [1, 2, 3, 4, 5, 6, 7, 10, 20, 40, 200]  # 8, 9,  10, 20, 40, 60, 80, 100, 140, 200]
    correctness = [[] for _ in xrange(len(test_data[0]))]
    for k in xrange(len(epochs)):
        print('###->', epochs[k], 'epochs')
        model0, model_name0 = mt.train2(m, tr_data, val_data, epochs[k], False,
                                        'cifar10_0445_epochsacc-5_', path=h5_path)
        acc, predicted_classes, _ = dt.predict_and_acc(model0, test_data)
        for c in xrange(len(correctness)):
            if predicted_classes[c] == test_data[1][c]:
                correctness[c].append(1)
            else:
                correctness[c].append(0)

        print('Test accuracy = ', acc)

    easy_imgs = []
    hard_imgs = []
    correctness_tot = [np.sum(img_preds) for img_preds in correctness]
    for c, n in enumerate(correctness_tot):
        if n == len(epochs):
            easy_imgs.append(c)
        if n == 0:
            hard_imgs.append(c)

    unique, counts = np.unique(correctness_tot, return_counts=True)
    n_correct = dict(zip(unique, counts))

    correctness_shapes = [str(img_preds) for img_preds in correctness]
    unique, counts = np.unique(correctness_shapes, return_counts=True)
    correct_shapes = dict(zip(unique, counts))
    sorted_cs = sorted(correct_shapes.items(), key=operator.itemgetter(1))
    print(n_correct)
    print(sorted_cs[-20:])

    print('Easy images ids: ', easy_imgs[max(-len(easy_imgs), -10):])
    print('Hard images ids: ', hard_imgs[max(-len(hard_imgs), -10):])


def colorcube_analysis():
    # m = 'densenet121'
    for m in models:
        test_data = dt.get_data('cifar10', (50000, 60000))
        top_n = 2500
        # model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
        model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False, suffix='ft20ep-exp')
        model = mt.load_by_name(model_name0, test_data[0].shape[1:], h5_path+model_name0)
        y_predicted = model.predict(np.array(test_data[0]))
        # y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
        true_classes = [int(k) for k in test_data[1]]
        scores = metrics.prediction_ratings(y_predicted, true_classes)
        score_sorted_ids = np.argsort(scores)
        cc_high = metrics_color.ColorDensityCube(resolution=4)
        for img_id in score_sorted_ids[-top_n:]:
            cc_high.feed(test_data[0][img_id])
        cc_high.normalize()
        cc_high.plot_cube()

        cc_low = metrics_color.ColorDensityCube(resolution=4)
        for img_id in score_sorted_ids[:top_n]:
            cc_low.feed(test_data[0][img_id])
        cc_low.normalize()

        cc_diff = cc_high.substract(cc_low, 'norm')

        cc_low.plot_cube()

        # cc_diff.normalize()
        cc_diff.plot_cube(title='Color cube analysis difference (' + str(top_n) + ' images/series)', normalize=False,
                          save=True)


def histogram_analysis():
    m = 'densenet121'
    test_data = dt.get_data('cifar10', (50000, 60000))
    top_n = 1000
    model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    true_classes = [int(k) for k in test_data[1]]
    scores = metrics.prediction_ratings(y_predicted, true_classes)
    score_sorted_ids = np.argsort(scores)
    high_score_series = []
    low_score_series = []
    for k in xrange(0, top_n):
        high_score_series.append(test_data[0][score_sorted_ids[-k-1]])
        low_score_series.append(test_data[0][score_sorted_ids[k]])

    plotting.plot_hists(high_score_series, 'high scores', low_score_series, 'low scores', plotting.cs_bgr,
                        title='Histogram analysis (' + str(top_n) + ' images/series)')


def show_ids():
    test_data = dt.get_data('cifar10', (50000, 60000))
    hard = [9746, 9840, 9853, 9901, 9910, 9923, 9924, 9926, 9960, 9982]
    easy = [9929, 9935, 9939, 9945, 9952, 9966, 9971, 9992, 9997, 9999]
    for k in easy:
        plotting.imshow(test_data[0][k])
    for k in hard:
        plotting.imshow(test_data[0][k])

    print('done')


def check_entropy():
    r_col_imgs = []
    r_bw_imgs = []
    for k in xrange(250, 0, -50):
        r_col_img = np.random.randint(k, 255, (32, 32, 3), np.uint8)
        r_bw_img = np.array([r_col_img[:, :, 0], r_col_img[:, :, 0], r_col_img[:, :, 0]], dtype=np.uint8)
        r_bw_img = np.swapaxes(r_bw_img, 0, 2)

        r_col_imgs.append(r_col_img)
        r_bw_imgs.append(r_bw_img)
        print(r_bw_img.shape)
        print('entropy:', metrics_color.entropy_cc(r_bw_img))
        # plotting.imshow(r_bw_img)  #, title='entropy_bw_'+str(k))
        print('entropy_cc:', metrics_color.entropy_cc(r_col_img))
        # plotting.imshow(r_col_img)  #, title='entropy_col_'+str(k))

    # rand_img = np.random.randint(0, 255, (32, 32, 3), np.uint8)
    # print('entropy random', metrics_color.entropy(rand_img))
    # plotting.imshow(rand_img)


def check_acc():
    m = 'densenet121'
    test_data = dt.get_data('cifar10', (50000, 60000))

    model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    predicted_classes = np.argmax(y_predicted, axis=1)
    print(predicted_classes[:10])
    true_classes = [int(k) for k in test_data[1]]
    acc = metrics.accuracy(predicted_classes, true_classes)
    print(acc)


def check_pr():
    m='densenet121'
    model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)

    test_data = dt.get_data('cifar10', (50000, 60000))
    easy = [9929, 9935, 9939, 9945, 9952, 9966, 9971, 9992, 9997, 9999]
    hard = [9746, 9840, 9853, 9901, 9910, 9923, 9924, 9926, 9960, 9982]
    cat = [671]
    cars = [6983, 3678, 3170, 1591]
    # plotting.show_imgs(easy, 'easy set: ', test_data[0], showColorCube=True, resolution=4)
    # plotting.show_imgs(hard, 'hard set: ', test_data[0], showColorCube=True, resolution=4)
    true_classes = [int(k) for k in test_data[1]]

    scores = metrics.prediction_ratings(y_predicted, true_classes)
    score_sorted_ids = np.argsort(scores)

    # print(scores[score_sorted_ids[0]], y_predicted[score_sorted_ids[0]])
    # print(scores[score_sorted_ids[1]], y_predicted[score_sorted_ids[1]])
    print(scores[score_sorted_ids[2500]], y_predicted[score_sorted_ids[2500]])
    print(scores[score_sorted_ids[2501]], y_predicted[score_sorted_ids[2501]])
    # print(scores[score_sorted_ids[9998]], y_predicted[score_sorted_ids[9998]])
    # print(scores[score_sorted_ids[9999]], y_predicted[score_sorted_ids[9999]])

    print('easy')
    for id in easy:
        print(id, '- pr:', metrics.prediction_rating(y_predicted[id], true_classes[id]),
              ' - correct?: ', np.argmax(y_predicted[id]) == true_classes[id])
        # print(y_predicted[id])
    print('hard')
    for id in hard:
        print(id, '- pr:', metrics.prediction_rating(y_predicted[id], true_classes[id]),
              ' - correct?: ', np.argmax(y_predicted[id]) == true_classes[id])
        # print(y_predicted[id])


def entropy_cc_analysis():
    m = 'densenet121'
    test_data = dt.get_data('cifar10', (50000, 60000))
    top_n = 2500

    model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    true_classes = [int(k) for k in test_data[1]]
    scores = metrics.prediction_ratings(y_predicted, true_classes)
    score_sorted_ids = np.argsort(scores)
    high_score_entropies = []
    low_score_entropies = []
    print(len(score_sorted_ids))
    for k in xrange(0, top_n):
        # id = score_sorted_ids[-k - 1]
        # print(id)
        # img = test_data[id]
        high_score_entropies.append(metrics_color.entropy_cc(test_data[0][score_sorted_ids[-k-1]], 8))
        low_score_entropies.append(metrics_color.entropy_cc(test_data[0][score_sorted_ids[k]], 8))

    plotting.box_plot(high_score_entropies, low_score_entropies, name_s1='high prediction scores',
                      name_s2='low prediction scores',y_label='Color cube entropy',
                      title='Entropy analysis (' + str(top_n) + ' images/series)')


def colorfulness_analysis(model='densenet121', top_n=2500):
    """
    Experiment to analyse the relevance if the colorfulness attribute
    See the metrics_color.colorfulness() function for more details on the attribute
    :param model: The predictions of :model: will be used to compute the prediciton scores
    :param top_n: Number of elements in the series that will be plotted for analysis
    :return:
    """

    # Load test data and model results
    test_data = dt.get_data('cifar10', (50000, 60000))
    model_name0 = mt.weight_file_name(model, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    true_classes = [int(k) for k in test_data[1]]

    # Compute scores and sort test data ids by score
    scores = metrics.prediction_ratings(y_predicted, true_classes)
    score_sorted_ids = np.argsort(scores)

    # Compute metric for high score and low score data
    high_score_series = []
    low_score_series = []
    print(len(score_sorted_ids))
    for k in xrange(0, top_n):
        high_score_series.append(metrics_color.colorfulness(test_data[0][score_sorted_ids[-k-1]]))
        low_score_series.append(metrics_color.colorfulness(test_data[0][score_sorted_ids[k]]))

    # Plot box plot of the two series
    plotting.box_plot(high_score_series, low_score_series, name_s1='high prediction scores',
                      name_s2='low prediction scores', y_label='Colorfulness',
                      title='Colorfulness analysis (' + str(top_n) + ' images/series)')


def check_rgb():
    test_data = dt.get_data('cifar10', (50000, 60000))
    # plotting.imshow(test_data[0][9960])
    # img_test = np.repeat(test_data[0][9960][:, :, 0, np.newaxis], 3, axis=2)
    img_test = np.array(test_data[0][9960])
    img_test[:, :, 1] = np.ones((32, 32)) #* 255
    img_test[:, :, 2] = np.ones((32, 32)) #* 255
    # img_test = np.swapaxes(img_test, 0, 2)
    print(np.array(test_data[0][9960]).shape)
    print(img_test)
    plotting.imshow(img_test)
    plotting.plot_hists([test_data[0][9960]], 'normal', [img_test], 'red', plotting.cs_bgr, )

def car_example():
    test_data = dt.get_data('cifar10', (50000, 60000))
    cars = [6983, 3678, 3170, 1591]

    cc0 = metrics_color.ColorDensityCube(resolution=4)
    cc0.feed(test_data[0][cars[0]])
    plotting.imshow(test_data[0][cars[0]])
    cc0.plot_cube()

    cc0 = metrics_color.ColorDensityCube(resolution=4)
    cc0.feed(test_data[0][cars[1]])
    plotting.imshow(test_data[0][cars[1]])
    cc0.plot_cube()


def show_distribution():
    images_cube = dt.cifar10_maxcolor_domains(granularity=4, data_range=(50000, 60000))
    region_sizes = dt.cube_cardinals(images_cube)
    cc = metrics_color.ColorDensityCube(resolution=4, cube=region_sizes)
    cc.normalize()
    cc.plot_cube()


def confusion(model='densenet121'):
    # Load test data and model results
    test_data = dt.get_data('cifar10', (50000, 60000))
    model_name0 = mt.weight_file_name(model, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    predicted_classes = np.argmax(y_predicted, axis=1)
    true_classes = [int(k) for k in test_data[1]]

    print('Confusion Matrix for Total Test Data')
    print(sk_metrics.confusion_matrix(true_classes, predicted_classes))

    scores = metrics.prediction_ratings(y_predicted, true_classes)
    prediction_scores = np.zeros((10, 1)).tolist()
    print(prediction_scores)
    for k in xrange(len(y_predicted)):
        prediction_scores[predicted_classes[k]].append(scores[k])

    print(np.array(prediction_scores).shape)
    for cifar_class in prediction_scores:
        print(float(np.mean(cifar_class)))


def retinanet_training_test():
    val_json = 'bdd100k_labels_images_val.json'
    train_json = 'bdd100k_labels_images_train.json'
    val_annot = 'val_annotations.csv'
    train_annot = 'train_annotations.csv'
    cl_map = 'class_mapping.csv'

    test_weight_file = 'test'

    classes = bu.annotate4retinanet(val_json, val_annot, bdd100k_labels_path, bdd100k_val_path,
                                    make_class_mapping=True, cl_map_file=cl_map)
    # bu.annotate4retinanet(val_json, val_annot, bdd100k_labels_path, bdd100k_val_path)
    bu.annotate4retinanet(train_json, train_annot, bdd100k_labels_path, bdd100k_train_path)
    # Hyper-parameters
    batch_size = 32

    for m in models:
        print('Generating %s backbone...' % m)
        backbone = kr_models.backbone(m)
        weights = backbone.download_imagenet()
        print('Creating generators...')
        tr_gen, val_gen = bu.create_generators(train_annotations=bdd100k_labels_path+train_annot,
                                               val_annotations=bdd100k_labels_path+val_annot,
                                               class_mapping=bdd100k_labels_path+cl_map,
                                               preprocess_image=backbone.preprocess_image)
        print('Creating models...')
        model, training_model, prediction_model = kr_train.create_models(backbone.retinanet, tr_gen.num_classes(), weights)
        print('Creating callbacks...')
        callbacks = bu.create_callbacks(model, batch_size, 'test', tensorboard_dir=tb_path)

        print('Training...')
        training_model.fit_generator(
            generator=tr_gen,
            steps_per_epoch=10000,  # 10000,
            epochs=2,
            verbose=2,
            callbacks=callbacks,
            workers=4,  # 1
            use_multiprocessing=True,  # False,
            max_queue_size=10,
            validation_data=val_gen
        )




def main():
    check_dirs(res_path,
               ilsvrc2012_path,
               h5_path,
               csv_path,
               png_path,
               bdd100k_labels_path,
               bdd100k_data_path,
               bdd100k_val_path,
               bdd100k_train_path)

    ### Experiments
    # imagenet_test()
    # finetune_test()
    # data_analysis()
    # bug_feature_detection()
    # color_domain()
    # cifar_color_domains_test()
    # color_region_finetuning()
    # mt_noise_test()
    # epochs_accuracy_test()
    # pr_on_fair_distribution()
    # cifar10_global_cc()


    ### Metric checks
    # check_entropy()
    # check_pr()
    # check_acc()
    # check_rgb()

    ### Attribute analysis
    # colorcube_analysis()
    # histogram_analysis()
    # entropy_cc_analysis()
    # colorfulness_analysis()

    ### Debugging
    # show_ids()

    ### Tests
    # test()
    # confusion()
    # show_distribution()
    retinanet_training_test()

main()
