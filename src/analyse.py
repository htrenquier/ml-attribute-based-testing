import attribute_analyser as aa
import operator
import model_trainer as mt
from keras.datasets import cifar10
import numpy as np

orig_train_data, orig_test_data = cifar10.load_data()
train_data, test_data = mt.format_data(orig_train_data, orig_test_data, 10)

imagenet_gt_file = '../ilsvrc2012/val_ground_truth.txt'

files = ('densenet121_cifar10_50ep_wda-res.csv', 'resnet50_cifar10_50ep_wda-res.csv',
         'mobilenet_cifar10_50ep_wda-res.csv',  # 'vgg16_cifar10_50ep_wda-res.csv',
         'mobilenetv2_cifar10_50ep_wda-res.csv',  # 'vgg19_cifar10_50ep_wda-res.csv',
         'nasnet_cifar10_50ep_wda-res.csv')

files_imagenet = ('densenet121-res.csv', 'mobilenet-res.csv', 'mobilenetv2-res.csv',
                  'nasnet-res.csv', 'resnet50-res.csv')  # , 'vgg16_imagenet-res.csv', 'vgg19_imagenet-res.csv')

res_path ='/Users/user/ws/ml-attribute-based-testing/res/'
# res_path ='/home/henri/ml-attribute-based-testing/res/'
images_path = '/home/henri/Downloads/imagenet-val/'

# preds16 = aa.load_csv('/Users/user/ws/ml-attribute-based-testing/src/vgg16_cifar10_50ep_wda-res.csv', 1)
# preds19 = aa.load_csv('/Users/user/ws/ml-attribute-based-testing/src/vgg19_cifar10_50ep_wda-res.csv', 1)
#
# print(np.count_nonzero(np.subtract(preds16, preds19)))


def conf_diff():
    # high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 4)
    # aa.plot_delta(aa.get_images(high_c, orig_test_data[0]),
    #               aa.get_images(low_c, orig_test_data[0]), aa.cs_bgr)
    # r_test = [np.random.random((32, 32, 3))*256 for _ in xrange(200)]
    # r_test = np.swapaxes([ np.ones((32, 32))*255,np.ones((32, 32))*255, np.zeros((32, 32))], 0, 2)
    # r_test = [r_test.tolist() for _ in xrange(2)]

    pr = aa.prediction_ratings(preds, true_classes)
    high_pr, low_pr = aa.sort_by_confidence(pr, len(pr) // 4)

    cdc_high = aa.ColorDensityCube(resolution=8)
    print(len(orig_train_data[0]))
    for img in aa.get_images(high_pr, orig_test_data[0]):
        cdc_high.feed(img)
    # cdc_train.avg()
    cdc_high.normalize()
    cdc_high.plot_cube()   # save=True, title='cifar_image_cube'+str(i))

    cdc_low = aa.ColorDensityCube(resolution=8)
    print(len(orig_train_data[0]))
    for img in aa.get_images(low_pr, orig_test_data[0]):
        cdc_low.feed(img)
    # cdc_train.avg()
    cdc_low.normalize()
    # cdc_low.plot_cube()  # save=True, title='cifar_image_cube'+str(i))

    cdc_diff = cdc_high.substract(cdc_low, state='avg')  # What does high has more than low?
    # cdc_diff.plot_cube()

    cdc_diff = cdc_high.substract(cdc_low, state='norm')  # What does high has more than low?
    # cdc_diff.plot_cube()

    cdc_finetune = aa.ColorDensityCube(resolution=8)
    finetune_data = aa.get_best_scores(orig_train_data[0], 1000, cdc_diff)
    for img_index in finetune_data:
        cdc_finetune.feed(orig_train_data[0][img_index])
    cdc_finetune.normalize()
    cdc_finetune.plot_cube()

    # cdc_high = aa.ColorDensityCube(resolution=4)
    # for img in aa.get_images(high_pr, orig_test_data[0]):
    #     cdc_high.color_cube(img)
    # cdc_high.normalize()
    # cdc_high.plot_cube()  # save=True, title='cifar_image_cube'+str(i))
    # cdc_low = aa.ColorDensityCube(resolution=4)
    # for img in aa.get_images(low_pr, orig_test_data[0]):
    #     cdc_low.color_cube(img)
    # cdc_low.normalize()
    # cdc_low.plot_cube()  # save=True, title='cifar_image_cube'+str(i))
    # cdc_diff = cdc_high.substract(cdc_low)
    # cdc_diff.plot_cube()




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
                           title='Top and bot 25% confidence for ' + file.split('_')[0],
                           path=images_path)
    high_c, low_c = aa.sort_by_confidence(confs, len(confs) / 8)
    image_ids_high_c = operator.itemgetter(*high_c)(filenames)
    image_ids_low_c = operator.itemgetter(*low_c)(filenames)
    aa.plot_hists_imagenet(image_ids_high_c, 'high conf',
                           image_ids_low_c, 'low conf', aa.cs_bgr,
                           title='Top and bot 12.5% confidence for ' + file.split('_')[0],
                           path=images_path)


for file in files[:1]:
    file_name = res_path + file
    # confs = aa.load_csv(file_name, 1)  # confidences
    # pred_class = aa.load_csv(file_name,2)
    preds = aa.load_csv(file_name, 3)  # predictions
    # try:
    #     assert(len(preds) == len(test_data[1]))
    # except AssertionError:
    #     print(len(preds))
    #     print(AssertionError)

    true_classes = np.argmax(test_data[1], axis=1)  # cifar
    # filenames, true_classes = aa.read_ground_truth(imagenet_gt_file)  # imagenet

    print(file.split('_')[0])
    # acc = aa.accuracy(preds, true_classes)

    ci = []  # confidence_incorrect = []
    ci_index = []
    cc = []  # confidence_correct = []
    cc_index = []

    # for i in xrange(len(preds)):
    #     if preds[i] == true_classes[i]:
    #         cc.append(confs[i])
    #     else:
    #         ci.append(confs[i])
    # aa.plot_conf_box(cc, ci, 'Correct(1) and Incorrect(2) confidences distribution for ' + file.split('_')[0])
    # correct_images, incorrect_images = aa.sort_by_correctness(preds, true_classes, orig_test_data[0])
    conf_diff()
    # imagenet_analysis()



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

# imagenet_analysis()
