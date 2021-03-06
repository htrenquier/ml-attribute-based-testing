from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.transforms as transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from keras.preprocessing import image
import numpy as np
import cv2
import metrics_color
import sklearn.preprocessing

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
    # plt.savefig(title+'.png')
    plt.show()
    plt.close()


def delta_hist(images1, images2):
    delta = [avg_hist(images2, 0) - avg_hist(images1, 0),
             avg_hist(images2, 1) - avg_hist(images1, 1),
             avg_hist(images2, 2) - avg_hist(images1, 2)]
    return delta


def color_3channels_hist(image_file):
    image = cv2.imread(image_file)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    # plt.title("'Color image's Color Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Amount of pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # plot the histogram
        # plt.legend(loc='upper right', shadow=True, fontsize='medium')
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
    plt.close


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


def box_plot(series1, series2, name_s1='series1', name_s2='series2', y_label=None, save=False, title=None):
    data = [series1, series2]
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels([name_s1, name_s2])  # rotation=45, fontsize=8)
    ax.set_ylabel(y_label)
    plt.show()
    if save:
        plt.savefig(title + '.png')
    plt.close()


def n_box_plot(series, series_names, y_label=None, save=False, title=None, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.boxplot(series, showfliers=False)
    ax.set_xticklabels(series_names, rotation=25, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_ylim([0.95, 1.001])
    if not ax:
        plt.show()
        if save:
            plt.savefig(title + '.png')
        plt.close()


def n_box_plot_compare(series, ref_series, series_names, y_label=None, save=False, title=None):
    def setBoxColors(bp):
        plt.setp(bp['boxes'][0], color='black')
        plt.setp(bp['caps'][0], color='black')
        plt.setp(bp['caps'][1], color='black')
        plt.setp(bp['whiskers'][0], color='black')
        plt.setp(bp['whiskers'][1], color='black')
        plt.setp(bp['medians'][0], color='black')

        plt.setp(bp['boxes'][1], color='grey')
        plt.setp(bp['caps'][2], color='grey')
        plt.setp(bp['caps'][3], color='grey')
        plt.setp(bp['whiskers'][2], color='grey')
        plt.setp(bp['whiskers'][3], color='grey')
        plt.setp(bp['medians'][1], color='grey')

    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    # plt.hold(True)

    # (r, g, b, a) = (0.8, 0.8, 0.8, 0.8)
    pos = 0
    ticks = [0]
    series_names.insert(0, '')
    for k in xrange(len(series)):
        bp = plt.boxplot([series[k], ref_series[k]], positions=[pos+1,pos+2], widths=0.6, showfliers=False)
        setBoxColors(bp)
        ticks.append(pos+1.5)
        pos += 3

    ax.set_xticks(ticks)
    ax.set_xticklabels(series_names, rotation=25, fontsize=12)
    ax.set_ylabel(y_label)
    ax.set_ylim([0.95, 1.001])

    hB, = plt.plot([1, 1], color='black', marker='_')
    hR, = plt.plot([1, 1], color='grey', marker='_')
    plt.legend((hB, hR), ('Fine-tuned', 'Reference'), loc='lower right')
    hB.set_visible(False)
    hR.set_visible(False)

    plt.show()
    if save:
        plt.savefig(title + '.png')
    plt.close()


def plot_discrete_attribute_scores(attributes, metric, title):
    fig, axs = plt.subplots(len(attributes), figsize=(12, 8*len(attributes)))
    fig.subplots_adjust(top=0.9)
    for ca, attribute in enumerate(attributes.values()):
        labels = attribute['d_attribute'].get_labels()
        distrib = [int(v) for v in attribute['d_attribute'].get_distribution()]
        series = [attribute['d_attribute'].get_metric_value_list(metric, label) for label in labels]
        series_names = ["%s (%1.2f)" % (labels[k], distrib[k]/sum(distrib)) for k in xrange(len(labels))]
        n_box_plot(series, series_names, metric, title='', ax=axs[ca])  # title=attribute['name'], ax=axs[ca])

    # fig.suptitle(title, x=0.80, y=0.997, fontsize=8)
    plt.show()
    plt.close()


def scale3d(ax, x_scale=1, y_scale=1, z_scale=1):
    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0
    short_proj = np.dot(Axes3D.get_proj(ax), scale)
    ax.get_proj = short_proj


def plot_cube(color_cube, fig=None, save=False, title=None, path='', normalize=True):
    assert isinstance(color_cube, metrics_color.ColorDensityCube)
    win = color_cube.get_win()
    res = color_cube.get_res()
    fig_was_none = False

    if not fig:
        fig_was_none = True
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(212, projection='3d')

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.8, 1.2, 0.8, 1.1]))  # burst view along y-axis
    ax.view_init(azim=-25)

    half_win_size = win//2
    print(half_win_size)
    axis = xrange(half_win_size, 256, win)
    for x in axis:
        for y in axis:
            if normalize:
                size = color_cube.get_normalized()[int(x / win)][int(y / win)] * 5000 / res
            else:
                size = color_cube.get_cube()[int(x / win)][int(y / win)] * 5000 / res
                print(size)
            color = [np.repeat(x/256, res),
                     np.repeat(y/256, res),
                     np.array(xrange(half_win_size, 256, win)) / 256.0]
            color = np.swapaxes(color, 0, 1)
            ec = np.where(size >= 0.0, 'w', 'r')
            size = abs(size)
            ax.scatter(x, y, axis, c=color, s=size, edgecolor=ec, alpha=1)

    if fig_was_none:
        plt.tight_layout()
        fig.text(0.5, 0.975, title, ha='center')
        plt.show()
    else:
        return
    if save:
        assert title is not None
        fig.text(0.5, 0.975, title, ha='center')
        plt.savefig(path + title + '.png')
    plt.close()


def show_imgs(id_list, title, dataset, tag_list=None, showColorCube=False, resolution=16):  # list of img list
        n_images = min(10, len(id_list))
        if n_images >= 1:
            fig, axes = plt.subplots(1 + showColorCube, n_images, squeeze=False, figsize=(n_images, 12),
                                     subplot_kw={'xticks': (), 'yticks': ()})

            if showColorCube:
                cc = metrics_color.ColorDensityCube(resolution=resolution)
                for i in xrange(n_images):
                    ax = axes[0][i]
                    img_id = id_list[i]
                    cc.feed(dataset[img_id])
                    ax.imshow(dataset[img_id], vmin=0, vmax=1)
                    if tag_list:
                        ax.set_title(tag_list[i])
                        # ax.set_title('label #' + str(id_list) + ' (' + str(i) + '/' + str(len(id_list)) + ' images)')
                plot_cube(cc, fig)
            else:
                for i in xrange(n_images):
                    ax = axes[0][i]
                    img_id = id_list[i]
                    ax.imshow(dataset[img_id], vmin=0, vmax=1)
                    if tag_list:
                        ax.set_title(tag_list[i])
                        # ax.set_title('label #' + str(id_list) + ' (' + str(i) + '/' + str(len(id_list)) + ' images)')
            if title:
                fig.suptitle(title + ' - label #' + str(id_list))
            # fig.subplots_adjust()
            plt.show()
        plt.close()


def plot_history(history, metric_name='acc', title='model'):
    # Plot training & validation accuracy values
    plt.plot(history.history[metric_name])
    plt.plot(history.history['val_'+metric_name])
    plt.title(title)
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()


