import attribute_analyser as aa
import model_trainer as mt
from keras.datasets import cifar10
import numpy as np
import cv2

orig_train_data, orig_test_data = cifar10.load_data()
train_data, test_data = mt.format_data(orig_train_data, orig_test_data, 10)

FILENAME = '/Users/user/ws/densenet121_cifar10_2ep_woda-res.csv'
res = aa.load_csv(FILENAME)
print(len(res))

assert(len(res) == len(test_data[1]))
acc = aa.accuracy(res, test_data[1])

true_classes = np.argmax(test_data[1], axis=1)


correct_images = []
incorrect_images = []

for i in xrange(len(res)):
    img = cv2.cvtColor(orig_test_data[0][i], cv2.COLOR_RGB2BGR)
    if res[i] == true_classes[i]:
        correct_images.append(img)
    else:
        incorrect_images.append(img)

aa.plot_hists(correct_images, 'correct', incorrect_images, 'incorrect', aa.cs_bgr)