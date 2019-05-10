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
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import math

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


def sigmoid(x):
    return 1/(1 + math.exp(-x))


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
    plt.close()


class ColorDensityCube:
    def __init__(self, resolution=16, cube=None):
        assert resolution and not resolution & (resolution - 1)  # resolution is a power of 2
        self.res = resolution
        if cube == None:
            self.cube = np.zeros((resolution, resolution, resolution))
        else:
            self.cube = cube
            self.res = len(cube)
        self.win = int(256/resolution)
        self.isNormalized = False
        self.norm_cube = np.zeros((resolution, resolution, resolution))
        self.num = 0

    def __getitem__(self, key):
        return self.cube[key]

    def __setitem__(self, key, value):
        self.cube[key] = value

    def feed(self, image):
        for x in image:
            for y in x:
                c0 = int(y[0]/self.win)
                c1 = int(y[1]/self.win)
                c2 = int(y[2]/self.win)
                self.cube[c0][c1][c2] += 1
        self.num += 1
        self.isNormalized = False

    def avg(self):
        return self.cube / self.num

    def count(self):
        count = 0
        cube = self.avg()
        for i in cube:
            for j in i:
                for k in j:
                    count += k
        return count

    def normalize(self):
        if self.num != 0:
            self.norm_cube = self.avg()
        else:
            self.norm_cube = self.cube
        max, min = abs(self.norm_cube).max(), 0  # abs(self.norm_cube).min()
        if not max:
            print('Cube is null')
            print(self.norm_cube)
        self.norm_cube = (self.norm_cube - min) / (max - min)
        self.isNormalized = True

    def substract(self, cube, state='avg'):
        diff_cube = ColorDensityCube(self.res)
        assert isinstance(cube, ColorDensityCube)
        for x in xrange(len(self.cube)):
            if state == 'avg':
                assert self.get_num() and cube.get_num()
                num = np.subtract(self.avg()[x], cube.avg()[x])
            elif state == 'norm':
                num = np.subtract(self.get_normalized()[x], cube.get_normalized()[x])
            elif state == 'value':
                assert self.get_num() == cube.get_num()
                num = np.subtract(self.get_cube()[x], cube.get_cube()[x])
            diff_cube[x] = num
            diff_cube.num = 1
        diff_cube.normalize()
        return diff_cube

    def get_normalized(self):
        if not self.isNormalized:
            self.normalize()
        return self.norm_cube

    def get_cube(self):
        return self.cube

    def get_num(self):
        return self.num

    def get_res(self):
        return self.res

    def plot_cube(self, save=False, title=None, path=''):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        axis = xrange(0, 256, self.win)
        for x in axis:
            for y in axis:
                size = self.norm_cube[int(x / self.win)][int(y / self.win)] * 10000 / self.res
                color = [np.repeat(x/256, self.res),
                         np.repeat(y/256, self.res),
                         np.array(xrange(int(self.win / 2), 256, self.win)) / 256.0]
                color = np.swapaxes(color, 0, 1)
                ec = np.where(size >= 0.0, 'w', 'r')
                size = abs(size)
                ax.scatter(x, y, axis, c=color, s=size, edgecolor=ec, alpha=1)
        #plt.show()
        if save:
            assert title is not None
            fig.text(0.5, 0.975, title, ha='center')
            plt.savefig(path + title + '.png')
        # plt.close()


def get_best_scores(images, num, diff_cube):
    assert isinstance(diff_cube, ColorDensityCube)
    scores = []
    for img in images:
        scores.append(cube_evaluate(img, diff_cube))
    args = np.argsort(scores)
    return args[-num:]


def cube_evaluate(img, diff_cube):
    assert isinstance(diff_cube, ColorDensityCube)
    cube = ColorDensityCube(diff_cube.get_res())
    cube.feed(img)
    score_cube = cube.substract(diff_cube, 'avg')
    return np.mean(score_cube.get_cube())


def evaluate_batch(images, diff_cube):
    assert isinstance(diff_cube, ColorDensityCube)
    cube = ColorDensityCube(diff_cube.get_res())
    for img in images:
        cube.feed(img)
    score_cube = cube.substract(diff_cube, 'avg')
    return np.mean(score_cube)


def colorfulness(image):
    # split the image into its respective RGB components
    # (B, G, R) = cv2.split(image.astype("float"))
    (R, G, B) = np.array(image[0])/255.0, np.array(image[1]/255.0), np.array(image[2]/255.0)

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def contrast(img):
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    m = np.mean(img_ycc[0])
    min, max = np.min(img_ycc[0]), np.max(img_ycc[0])
    return m/(max-min)

def finetune_by_colorfulness(ft_data_src, num, model_name0, res_path):
    col_scores =[]
    for img in ft_data_src:
        col_scores.append(colorfulness(img))
    col_scores = np.argsort(col_scores)
    return col_scores[:num]


def finetune_by_cdc(high_pr, low_pr, test_data_orig, ft_data_src, model_name, res_path):
    # ft_data_orig is the data to make a selection form

    # study test data color distrib
    cdc_high = ColorDensityCube(resolution=8)
    for img in get_images(high_pr, test_data_orig[0]):
        cdc_high.feed(img)
    cdc_high.normalize()
    cdc_high.plot_cube(save=True, title=model_name + '-high_pr', path=res_path)

    cdc_low = ColorDensityCube(resolution=8)
    for img in get_images(low_pr, test_data_orig[0]):
        cdc_low.feed(img)
    cdc_low.normalize()
    cdc_low.plot_cube(save=True, title=model_name + '-low_pr', path=res_path)

    cdc_diff = cdc_high.substract(cdc_low, state='norm')  # What does high has more than low?
    # cdc_diff.plot_cube()

    # Fine-tune data selection
    cdc_finetune = ColorDensityCube(resolution=8)
    finetune_data_args = get_best_scores(ft_data_src[0], 10000, cdc_diff)

    for img_index in finetune_data_args:
        cdc_finetune.feed(ft_data_src[0][img_index])
    cdc_finetune.normalize()
    cdc_finetune.plot_cube(save=True, title=model_name + '-ft_selection', path=res_path)

    return finetune_data_args



def imshow(img):
    plt.imshow(img)
    # plt.savefig(title + '.png')
    plt.show()


def plot(x, y, save=False, title=None):
    plt.plot(x, y, 'o')
    if save:
        assert title is not None
        plt.savefig(title + '.png')
    plt.show()

def avg_hist(images, channel):
    hist = np.zeros(256)
    # print(channel)
    for img in images:
        hist = hist + np.concatenate(cv2.calcHist([img], [channel], None, [256], [0, 256]))
    return hist/len(images)


def hist_conv(hist, win_size=10, stride=5):
    conv_hist = [[],[],[]]
    for c in xrange(len(hist)):
        for x in xrange(0, len(hist[c]), stride):
            print(conv_hist)
            if x + win_size >= len(hist):
                conv_hist[c].append(np.mean(hist[c][x:]))
            else:
                conv_hist[c].append(np.mean(hist[c][x:x + win_size]))
    return conv_hist


def plot_hists(images1, label1, images2, label2, color_space, title='Untitled plot'):
    fig, axs = plt.subplots(1, len(color_space), sharex='row')
    fig.text(0.005, 0.5, 'Number of pixels', va='center', rotation='vertical')
    fig.text(0.5, 0.975, title, ha='center')
    images1 = convert_cs(images1, color_space)
    images2 = convert_cs(images2, color_space)
    for j, ch in enumerate(color_space):
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
    plt.close()


def delta_hist(images1, images2):
    delta = [avg_hist(images2, 0) - avg_hist(images1, 0),
             avg_hist(images2, 1) - avg_hist(images1, 1),
             avg_hist(images2, 2) - avg_hist(images1, 2)]
    return delta


def plot_delta(images1, images2, color_space):
    images1 = convert_cs(images1, color_space)
    images2 = convert_cs(images2, color_space)
    win_size = 10
    stride = 5
    delta = hist_conv(delta_hist(images1, images2), win_size, stride)
    norm_delta = preprocessing.normalize(delta)
    # for j, ch in enumerate(color_space):
    #     norm_delta.append(preprocessing.normalize(delta[j]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = []
    sizes = []
    points_x = []
    points_y = []
    points_z = []
    for x in xrange(len(norm_delta[0])):
        for y in xrange(len(norm_delta[1])):
            for z in xrange(len(norm_delta[2])):
                xv = norm_delta[0][x]
                yv = norm_delta[1][y]
                zv = norm_delta[2][z]
                size = np.sqrt(xv**2 + yv**2 + zv**2)
                if size >= 0.2:
                    colors.append([float(min(x*stride, 256)/256), float(min(y*stride, 256)/256),
                                   float(min(z*stride, 256)/256)])
                    sizes.append(size*300)
                    points_x.append(x)
                    points_y.append(y)
                    points_z.append(z)
    ax.scatter(points_x, points_y, points_z, c=colors)  # , s=sizes)
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


def prediction_rating(prediction, true_class):
    p_true = prediction[true_class]
    prediction = np.delete(prediction, true_class)
    p_max, p_min = np.max(prediction), np.min(prediction)
    x = (1 + p_true - p_max) / (p_max - p_min)
    return math.atan(x)*2/math.pi


def prediction_ratings(predictions, true_classes):
    return [prediction_rating(predictions[i], true_classes[i]) for i in xrange(len(predictions))]


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
            infos = str.split(", ")
            if len(infos) > 1:
                if col == 1:
                    info.append(float(infos[col]))
                elif col == 3:
                    infos = infos[col].lstrip(' [').rstrip(']\r\n')
                    info.append([float(k) for k in infos.split()])
                else:
                    info.append(int(str[col]))

            str = ''
    f.close()
    return info


def accuracy(predicted_classes, true_classes):
    nz = np.count_nonzero(np.subtract(predicted_classes, true_classes))
    acc = (len(true_classes) - nz) / len(true_classes)
    print('Test Accuracy = ' + str(acc))
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

