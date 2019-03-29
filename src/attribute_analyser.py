from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Color-spaces
cs_bgr = ('Blue', 'Green', 'Red')
cs_hsv = ('Hue', 'Saturation', 'Value')
cs_ycrcb = ('Y (Luma)', 'Cr', 'Cb')
cs_lab = ('Lightness', 'a', 'b')
cs_grey_scale = ['Grey']


def avg_hist(images, channel):
    hist = np.zeros(256)
    print(channel)
    for img in images:
        hist = hist + np.concatenate(cv2.calcHist([img], [channel], None, [256], [0, 256]))
    return hist/len(images)


def plot_hists(images1, label1, images2, label2, color_space, title='Untitled plot'):
    fig, axs = plt.subplots(1, len(color_space), sharex='row')
    fig.text(0.005, 0.5, 'Number of pixels', va='center', rotation='vertical')
    fig.text(0.5, 0.975, title, ha='center')
    images1 = convert_cs(images1, color_space)
    images2 = convert_cs(images2, color_space)
    for j, ch in enumerate(color_space):
        print(j)
        print(ch)
        ax = axs[j]
        ax.plot(avg_hist(images1, j), label=label1, color='g')
        ax.plot(avg_hist(images2, j), label=label2, color='r')
        ax.set_title(ch + ' channel')
        ax.set_xlabel('Pixel values')
        # ax.set_ylabel('Number of pixels')
        ax.legend(loc='upper right', shadow=True, fontsize='medium')
    # fig.subplots_adjust(top=0.85)
    # fig.suptitle(title)
    # plt.savefig(title+'.png')
    plt.show()


def confidence(prediction):
    m = np.max(prediction)
    return m - (sum(prediction) - m) / (len(prediction) - 1)


def confidences(predictions):
    return [confidence(p) for p in predictions]


def sort_by_confidence(confidences, number_elements=None):
    """
    Crescent sort
    :param confidences: List of confidences
    :param number_elements: How many elements to return
    :return: Two lists of indexes for high and low confidences.
    """
    if number_elements is None or number_elements > len(confidences)//2:
        number_elements = len(confidences)//2
    sorted_args = np.argsort(confidences)
    # return high_confidence, low_confidence
    return sorted_args[-number_elements:], sorted_args[:number_elements]


def get_images(indexes, images):
    return [images[i] for i in indexes]


def load_csv(file_name, col):
    """
    Extracts info from csv file from test_bench
    :param file_name: file to get info from
    :param col: column in the csv file to get info from
            0: image id
            1: confidence
            2: predicted classes
            3: confidences (loss vector)
    :return: array of /col/ info
    """
    f = open(file_name, "r")
    info = []
    str = ''
    for l in f.readlines():
        str = str + l
        if ']' in str:
            str = str.split(", ")
            if len(str) > 1:
                if col == 1:
                    info.append(float(str[col]))
                elif col == 3:
                    str = str[col].lstrip('[').rsplit(']')
                    info.append(confidence([float(k) for k in str.split()]))
                else:
                    info.append(int(str[col]))

            str = ''
    f.close()
    return info


def accuracy(predicted_classes, y_test):
    true_classes = np.argmax(y_test, axis=1)
    nz = np.count_nonzero(np.subtract(predicted_classes, true_classes))
    acc = (len(y_test) - nz) / len(y_test)
    print('Accuracy = ' + str(acc))
    return acc


def convert_cs(images, cs):
    if cs == cs_bgr:
        return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
    elif cs == cs_hsv:
        return [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in images]
    elif cs == cs_lab:
        return [cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in images]
    elif cs == cs_ycrcb:
        return [cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB) for img in images]
    elif cs == cs_grey_scale:
        return [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    else:
        return images


def sort_by_correctness(predictions, true_classes, orig_images):
    correct_images = []
    incorrect_images = []
    for i in xrange(len(predictions)):
        if predictions[i] == true_classes[i]:
            correct_images.append(orig_images[i])
        else:
            incorrect_images.append(orig_images[i])
    return correct_images, incorrect_images