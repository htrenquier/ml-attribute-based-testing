import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import cv2

# Color-spaces
cs_bgr = ('Blue', 'Green', 'Red')
cs_hsv = ('Hue', 'Saturation', 'Value')
cs_ycrcb = ('Y (Luma)', 'Cr', 'Cb')
cs_lab = ('Lightness', 'a', 'b')
cs_grey_scale = ['Grey']


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


def avg_hist_imagenet(image_ids, channel, cs, path=''):
    """
    Returns an average histogram (pixel value distribution) for a list of image files
    :param image_ids: image file name
    :param channel: channel to study
    :param cs: target color space (input cs is asssumed to be RGB)
    :param path: path of the images (without image file name)
    :return: the average distribution of pixel of :channel: across all :image_ids:
    """
    hist = np.zeros(256)
    # print(channel)
    for id in image_ids:
        img = image.load_img(path+id, target_size=(224, 224))
        img = image.img_to_array(img)
        convert_one(img, cs)
        hist = hist + np.concatenate(cv2.calcHist([img], [channel], None, [256], [0, 256]))
    return hist / len(image_ids)


def plot_hists_imagenet(image_ids1, label1, image_ids2, label2, color_space, path, title='Untitled plot'):
    """
    Plots a comparaison per channel of :color_space: of the average histograms of sets of images :image_ids1: and
    :image_ids2:.
    :param image_ids1:
    :param label1:
    :param image_ids2:
    :param label2:
    :param color_space: color space of studied channels
    :param path: path of the images
    :param title: title of the plot.
    """
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


def imshow(img, title=None):
    plt.imshow(img)
    if title:
        plt.savefig(title + '.png')
    plt.show()


def quick_plot(x, y, title=None):
    plt.plot(x, y, 'o')
    if title:
        plt.savefig(title + '.png')
    plt.show()


def avg_hist(images, channel):
    hist = np.zeros(256)
    # print(channel)
    for img in images:
        hist = hist + np.concatenate(cv2.calcHist([img], [channel], None, [256], [0, 256]))
    return hist/len(images)


def hist_conv(hist, win_size=10, stride=5):
    conv_hist = [[], [], []]
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