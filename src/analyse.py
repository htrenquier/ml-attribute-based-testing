import attribute_analyser as aa
import operator
import model_trainer as mt
from keras.datasets import cifar10
import numpy as np

# orig_train_data, orig_test_data = cifar10.load_data()
# train_data, test_data = mt.format_data(orig_train_data, orig_test_data, 10)

imagenet_gt_file = '../ilsvrc2012/val_ground_truth.txt'

files = ('densenet121_cifar10_50ep_wda-res.csv', 'resnet50_cifar10_50ep_wda-res.csv',
         'mobilenet_cifar10_50ep_wda-res.csv', 'vgg16_cifar10_50ep_wda-res.csv',
         'mobilenetv2_cifar10_50ep_wda-res.csv', 'vgg19_cifar10_50ep_wda-res.csv',
         'nasnet_cifar10_50ep_wda-res.csv')

files_imagenet = ('densenet121-res.csv', 'mobilenet-res.csv', 'mobilenetv2-res.csv',
                  'nasnet-res.csv', 'resnet50-res.csv', 'vgg16_imagenet-res.csv', 'vgg19_imagenet-res.csv')

# path ='/Users/user/ws/ml-attribute-based-testing/res/'
res_path ='/home/henri/ml-attribute-based-testing/res/'
images_path = '/home/henri/Downloads/imagenet-val'

# preds16 = aa.load_csv('/Users/user/ws/ml-attribute-based-testing/src/vgg16_cifar10_50ep_wda-res.csv', 1)
# preds19 = aa.load_csv('/Users/user/ws/ml-attribute-based-testing/src/vgg19_cifar10_50ep_wda-res.csv', 1)
#
# print(np.count_nonzero(np.subtract(preds16, preds19)))

def cifar_analysis():
    high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 2)
    aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
                  aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_ycrcb,
                  title='Top and bot 50% confidence for ' + file.split('_')[0])

    high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 4)
    aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
                  aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_ycrcb,
                  title='Top and bot 25% confidence for ' + file.split('_')[0])

    high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 8)
    aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
                  aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_ycrcb,
                  title='Top and bot 12.5% confidence for ' + file.split('_')[0])


def imagenet_analysis():
    high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 2)
    image_ids_high_c = operator.itemgetter(*high_c)(filenames)
    image_ids_low_c = operator.itemgetter(*low_c)(filenames)
    aa.plot_hists_imagenet(image_ids_high_c, 'high conf',
                           image_ids_low_c, 'low conf', aa.cs_bgr,
                           title='Top and bot 50% confidence for ' + file.split('_')[0],
                           path=images_path)

    high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 4)
    image_ids_high_c = operator.itemgetter(*high_c)(filenames)
    image_ids_low_c = operator.itemgetter(*low_c)(filenames)
    aa.plot_hists_imagenet(image_ids_high_c, 'high conf',
                           image_ids_low_c, 'low conf', aa.cs_bgr,
                           title='Top and bot 50% confidence for ' + file.split('_')[0],
                           path=images_path)
    high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 8)
    image_ids_high_c = operator.itemgetter(*high_c)(filenames)
    image_ids_low_c = operator.itemgetter(*low_c)(filenames)
    aa.plot_hists_imagenet(image_ids_high_c, 'high conf',
                           image_ids_low_c, 'low conf', aa.cs_bgr,
                           title='Top and bot 50% confidence for ' + file.split('_')[0],
                           path=images_path)


for file in files:
    file_name = res_path + file
    confs = aa.load_csv(file_name, 1)  # confidences
    preds = aa.load_csv(file_name, 2)  # predictions
    # try:
    #     assert(len(preds) == len(test_data[1]))
    # except AssertionError:
    #     print(len(preds))
    #     print(AssertionError)

    # true_classes = np.argmax(test_data[1], axis=1)  # cifar
    filenames, true_classes = aa.read_ground_truth(imagenet_gt_file)  # imagenet
    print(file.split('_')[0])
    acc = aa.accuracy(preds, true_classes)

    ci = []  # confidence_incorrect = []
    ci_index = []
    cc = []  # confidence_correct = []
    cc_index = []

    for i in xrange(len(preds)):
        if preds[i] == true_classes[i]:
            cc.append(confs[i])
        else:
            ci.append(confs[i])
    aa.plot_conf_box(cc, ci, 'Correct(1) and Incorrect(2) confidences distribution for ' + file.split('_')[0])
    # correct_images, incorrect_images = aa.sort_by_correctness(preds, true_classes, orig_test_data[0])
    imagenet_analysis()



# aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
#               aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_hsv, title='Top and bot 50%')
# aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
#               aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_lab, title='Top and bot 50%')
# aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
#               aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_ycrcb, title='Top and bot 50%')

# aa.plot_hists(correct_images, 'correct', incorrect_images, 'incorrect', aa.cs_bgr)
# aa.plot_hists(correct_images, 'correct', incorrect_images, 'incorrect', aa.cs_hsv)
# aa.plot_hists(correct_images, 'correct', incorrect_images, 'incorrect', aa.cs_lab)
# aa.plot_hists(correct_images, 'correct', incorrect_images, 'incorrect', aa.cs_ycrcb)
# aa.plot_hists(correct_images, 'correct', incorrect_images, 'incorrect', aa.cs_grey_scale)

imagenet_analysis()
