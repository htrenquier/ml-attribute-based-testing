import numpy as np
import cv2
import matplotlib.pyplot as plt

# Color-spaces
cs_bgr = ('Blue', 'Green', 'Red')
cs_hsv = ('Hue', 'Saturation', 'Value')
cs_ycrcb = ('Y (Luma)', 'Cr', 'Cb')
cs_lab = ('Lightness', 'a', 'b')
cs_grey_scale = ('Grey')


def avg_hist(images, channel):
    hist = np.zeros(256)
    for img in images:
        hist = hist + np.concatenate(cv2.calcHist([img], [channel], None, [256], [0, 256]))
    return hist/len(images)


def plot_hists(images1, label1, images2, label2, color_space):
    fig, axs = plt.subplots(1, len(color_space), sharex='row')
    x = xrange(256)
    for j, ch in enumerate(color_space):
        ax = axs[j]
        ax.plot(avg_hist(images1, j), label=label1, color='g')
        ax.plot(avg_hist(images2, j), label=label2, color='r')
        #ax.set_xlim([0, 256])
        ax.set_title(ch + ' channel')
        ax.set_xlabel('Pixel values')
        ax.set_ylabel('Number of pixels')
        ax.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.show()


def confidence(prediction):
    m = np.max(prediction)
    return m - (sum(prediction) - m) / (len(prediction) - 1)



