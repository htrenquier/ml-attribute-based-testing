from __future__ import division
import numpy as np
import metrics
import metrics_color
import plotting
import model_trainer as mt
import data_tools as dt
import tests_logging as t_log
import cv2
import initialise

csv_path = '../res/csv/'

# models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50')
models = ['densenet121', 'resnet50']

def check_entropy():
    r_col_imgs = []
    r_bw_imgs = []
    test_data = dt.get_data('cifar10', (50000, 60000))
    entropies = []
    for img in test_data[0]:
        entropies.append(metrics_color.entropy_cc(img))

    sorted_args = np.argsort(entropies)

    plotting.imshow(test_data[0][sorted_args[0]])
    print(entropies[sorted_args[0]], test_data[1][sorted_args[0]])
    plotting.imshow(test_data[0][sorted_args[100]])
    print(entropies[sorted_args[100]], test_data[1][sorted_args[100]])
    plotting.imshow(test_data[0][sorted_args[1000]])
    print(entropies[sorted_args[1000]], test_data[1][sorted_args[1000]])
    plotting.imshow(test_data[0][sorted_args[9000]])
    print(entropies[sorted_args[9000]], test_data[1][sorted_args[9000]])
    plotting.imshow(test_data[0][sorted_args[9900]])
    print(entropies[sorted_args[9900]], test_data[1][sorted_args[9900]])
    plotting.imshow(test_data[0][sorted_args[9999]])
    print(entropies[sorted_args[9999]], test_data[1][sorted_args[9999]])

    # for k in xrange(250, 0, -50):
    #     r_col_img = np.random.randint(k, 255, (32, 32, 3), np.uint8)
    #     r_bw_img = np.array([r_col_img[:, :, 0], r_col_img[:, :, 0], r_col_img[:, :, 0]], dtype=np.uint8)
    #     r_bw_img = np.swapaxes(r_bw_img, 0, 2)
    #
    #     r_col_imgs.append(r_col_img)
    #     r_bw_imgs.append(r_bw_img)
    #     print(r_bw_img.shape)
    #     print('entropy:', metrics_color.entropy_cc(r_bw_img))
    #     # plotting.imshow(r_bw_img)  #, title='entropy_bw_'+str(k))
    #     print('entropy_cc:', metrics_color.entropy_cc(r_col_img))
    #     # plotting.imshow(r_col_img)  #, title='entropy_col_'+str(k))

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
    m = 'densenet121'
    model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)

    test_data = dt.get_data('cifar10', (50000, 60000))
    easy = [9929, 9935, 9939, 9945, 9952, 9966, 9971, 9992, 9997, 9999]
    hard = [9746, 9840, 9853, 9901, 9910, 9923, 9924, 9926, 9960, 9982]
    # cat = [671]
    # cars = [6983, 3678, 3170, 1591]
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
    for img_id in easy:
        print(img_id, '- pr:', metrics.prediction_rating(y_predicted[img_id], true_classes[img_id]),
              ' - correct?: ', np.argmax(y_predicted[img_id]) == true_classes[img_id])
        # print(y_predicted[id])
    print('hard')
    for img_id in hard:
        print(img_id, '- pr:', metrics.prediction_rating(y_predicted[img_id], true_classes[img_id]),
              ' - correct?: ', np.argmax(y_predicted[img_id]) == true_classes[img_id])
        # print(y_predicted[id])


def check_rgb():
    test_data = dt.get_data('cifar10', (50000, 60000))
    # plotting.imshow(test_data[0][9960])
    # img_test = np.repeat(test_data[0][9960][:, :, 0, np.newaxis], 3, axis=2)
    img_test = np.array(test_data[0][9960])
    img_test[:, :, 1] = np.ones((32, 32))  # * 255
    img_test[:, :, 2] = np.ones((32, 32))  # * 255
    # img_test = np.swapaxes(img_test, 0, 2)
    print(np.array(test_data[0][9960]).shape)
    print(img_test)
    plotting.imshow(img_test)
    plotting.plot_hists([test_data[0][9960]], 'normal', [img_test], 'red', plotting.cs_bgr, )


def check_bdd_extracted_dataset(labels):
    none_types = []
    wrong_sizes = []
    with open(labels, 'r') as fd:
        line = fd.readline()
        while line:
            id = line.split(',')[0]
            im = cv2.imread(id)
            if type(im) is np.ndarray:
                if im.shape != (64, 64, 3):
                    wrong_sizes.append(id)
            else:
                none_types.append(id)
            line = fd.readline()
    print(len(none_types))
    print(none_types[:10])
    print(len(wrong_sizes))
    print(wrong_sizes[:10])

def main():
    """
    Metric checks
    :return:
    """
    # check_entropy()
    # check_pr()
    # check_acc()
    # check_rgb()
    train_labels = '../../bdd100k/classification/labels/train_ground_truth.csv'
    val_labels = '../../bdd100k/classification/labels/val_ground_truth.csv'

    print("train data check")
    check_bdd_extracted_dataset(train_labels)
    print("val data check")
    check_bdd_extracted_dataset(val_labels)


main()