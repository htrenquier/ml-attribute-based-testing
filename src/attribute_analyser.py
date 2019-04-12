from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16

# Color-spaces
cs_bgr = ('Blue', 'Green', 'Red')
cs_hsv = ('Hue', 'Saturation', 'Value')
cs_ycrcb = ('Y (Luma)', 'Cr', 'Cb')
cs_lab = ('Lightness', 'a', 'b')
cs_grey_scale = ['Grey']


class ImagenetGenerator(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
            for file_name in batch_x]), np.array(batch_y)


def read_ground_truth(gt_file):
    # gt for groundtruth
    # file available @ http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    true_classes = []
    filenames = []
    with open(gt_file, 'r') as f:
        for l in f.readlines():
            line = l.split()
            filenames.append(line[0].strip())
            true_classes.append(int(line[1].strip()))
    return filenames, true_classes


def predict_batch(model, images):
    if images != []:
        y_predicted = model.predict(images)
        predicted_classes = np.argmax(y_predicted, axis=1)
        return predicted_classes.tolist()
    else:
        return []


def predict_dataset(filenames, path, model, model_preprocess_function):
    """
    For imagenet
    :param filenames: file of filenames
    :param model_preprocess_function:
    :param path: path of test images
    :param model:
    :return: predictions
    """
    y_predicted = []
    batch_size = 32
    batch = []
    for filename in filenames:
        batch.append(process(path+filename, model_preprocess_function))
        if len(batch) >= batch_size:
            y_predicted = y_predicted + model.predict(np.array(batch)).tolist()
            batch = []
    y_predicted = y_predicted + model.predict(np.array(batch)).tolist()
    return y_predicted


def process(file_path, model_preprocess_function):
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    x = model_preprocess_function(x)
    return x


def avg_hist_imagenet(image_ids, channel, cs, path=''):
    hist = np.zeros(256)
    # print(channel)
    for id in image_ids:
        img = image.load_img(path+id, target_size=(224, 224))
        img = image.img_to_array(img)
        convert_one(img, cs)
        hist = hist + np.concatenate(cv2.calcHist([img], [channel], None, [256], [0, 256]))
    return hist / len(image_ids)


def plot_hists_imagenet(image_ids1, label1, image_ids2, label2, color_space, path, title='Untitled plot'):
    fig, axs = plt.subplots(1, len(color_space), sharex='row')
    fig.text(0.005, 0.5, 'Number of pixels', va='center', rotation='vertical')
    fig.text(0.5, 0.975, title, ha='center')
    for j, ch in enumerate(color_space):
        print(j)
        print(ch)
        ax = axs[j]
        ax.plot(avg_hist_imagenet(image_ids1, j, color_space, path), label=label1, color='g')
        ax.plot(avg_hist_imagenet(image_ids2, j, color_space, path), label=label2, color='r')
        ax.set_title(ch + ' channel')
        ax.set_xlabel('Pixel values')
        # ax.set_ylabel('Number of pixels')
        ax.legend(loc='upper right', shadow=True, fontsize='medium')
    # fig.subplots_adjust(top=0.85)
    # fig.suptitle(title)
    plt.savefig(title+'.png')
    plt.show()


def avg_hist(images, channel):
    hist = np.zeros(256)
    # print(channel)
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
    plt.savefig(title+'.png')
    plt.show()


def plot_conf_box(cc, ci, title):
    data = [cc, ci]
    fig3, ax3 = plt.subplots()
    ax3.set_title(title)
    ax3.boxplot(data, showfliers=False)
    plt.savefig(title + '.png')
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


def accuracy(predicted_classes, true_classes):
    nz = np.count_nonzero(np.subtract(predicted_classes, true_classes))
    acc = (len(true_classes) - nz) / len(true_classes)
    print('Accuracy = ' + str(acc))
    return acc


def convert_one(image, cs):
    if cs == cs_bgr:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif cs == cs_hsv:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cs == cs_lab:
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    elif cs == cs_ycrcb:
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    elif cs == cs_grey_scale:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        return image


def convert_cs(images, cs):
        return [convert_one(img, cs) for img in images]


def sort_by_correctness(predictions, true_classes, orig_images):
    correct_images = []
    incorrect_images = []
    for i in xrange(len(predictions)):
        if predictions[i] == true_classes[i]:
            correct_images.append(orig_images[i])
        else:
            incorrect_images.append(orig_images[i])
    return correct_images, incorrect_images