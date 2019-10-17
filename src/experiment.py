from __future__ import division
import numpy as np
import operator
import metrics
import metrics_color
import plotting
import model_trainer as mt
import data_tools as dt
import initialise

# models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50')
models = ['densenet121', 'resnet50']

h5_path = '../res/h5/'


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
        model_base, model_name0 = mt.train2(m, tr_data, val_data, 50, False, 'cr_0245', path=h5_path)
        scores_cubes = []

        for x in xrange(g):
            nametag_prefix = 'ft_2345_ref' + str(x + 4)
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
                        print('#--> Region ' + str(x) + str(y) + str(z) + ' (' + str(
                            region_sizes[x][y][z]) + ' images)')
                        nametag_prefix = 'ft_2445_r' + str(x) + str(y) + str(z) + '_cr_1'

                        ft_model_name = mt.ft_weight_file_name(model_name0, ft_data_augmentation, ft_epochs,
                                                               nametag=nametag_prefix + 'exp')
                        weights_file = h5_path + ft_model_name + '.h5'

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


def mt_noise_test():
    np.random.seed(0)
    tr_data = dt.get_data('cifar10', (0, 40000))
    val_data = dt.get_data('cifar10', (40000, 50000))
    for noise_level in xrange(5, 200, 10):
        for k in [1]:  # xrange(len(tr_data[0])):
            # noise_mat = np.repeat(np.random.random((32, 32))[:, :, np.newaxis], 3, axis=2)
            noise_mat = np.swapaxes([np.random.random((32, 32)), np.random.random((32, 32)),
                                     np.random.random((32, 32))], 0, 2)
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


def main():

    """
    Experiments
    :return:
    """

    bug_feature_detection()
    color_region_finetuning()
    mt_noise_test()
    epochs_accuracy_test()

main()