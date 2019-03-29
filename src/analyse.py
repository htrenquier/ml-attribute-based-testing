import attribute_analyser as aa
import model_trainer as mt
from keras.datasets import cifar10
import numpy as np

orig_train_data, orig_test_data = cifar10.load_data()
train_data, test_data = mt.format_data(orig_train_data, orig_test_data, 10)

files = ('densenet121_cifar10_50ep_wda-res.csv', 'resnet50_cifar10_50ep_wda-res.csv',
         'mobilenet_cifar10_50ep_wda-res.csv', 'vgg16_cifar10_50ep_wda-res.csv',
         'mobilenetv2_cifar10_50ep_wda-res.csv', 'vgg19_cifar10_50ep_wda-res.csv',
         'nasnet_cifar10_50ep_wda-res.csv')
path ='/Users/user/ws/ml-attribute-based-testing/src/'


preds16 = aa.load_csv('/Users/user/ws/ml-attribute-based-testing/src/vgg16_cifar10_50ep_wda-res.csv', 1)
preds19 = aa.load_csv('/Users/user/ws/ml-attribute-based-testing/src/vgg19_cifar10_50ep_wda-res.csv', 1)

print(np.count_nonzero(np.subtract(preds16, preds19)))
#
# for file in files:
#     file_name = path + file
#     confs = aa.load_csv(file_name, 1)  # confidences
#     preds = aa.load_csv(file_name, 2)  # predictions
#     try:
#         assert(len(preds) == len(test_data[1]))
#     except AssertionError:
#         print(len(preds))
#         print(AssertionError)
#
#     acc = aa.accuracy(preds, test_data[1])
#     true_classes = np.argmax(test_data[1], axis=1)
#
#     correct_images, incorrect_images = aa.sort_by_correctness(preds, true_classes, orig_test_data[0])
#     high_c, low_c = aa.sort_by_confidence(confs, len(confs)/2)
#     aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
#                   aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_ycrcb,
#                   title='Top and bot 50% confidence for ' + file.split('_')[0])
#
#     high_c, low_c = aa.sort_by_confidence(confs, len(confs)/4)
#     aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
#                   aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_ycrcb,
#                   title='Top and bot 25% confidence for ' + file.split('_')[0])
#
#     high_c, low_c = aa.sort_by_confidence(confs, len(confs)/8)
#     aa.plot_hists(aa.get_images(high_c, orig_test_data[0]), 'high conf',
#                   aa.get_images(low_c, orig_test_data[0]), 'low conf', aa.cs_ycrcb,
#                   title='Top and bot 12.5% confidence for ' + file.split('_')[0])
#


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

