import matplotlib.pyplot as plt
import numpy as np
import metrics
import data_tools as dt
import cv2

# Color-spaces
cs_bgr = ('Blue', 'Green', 'Red')
cs_hsv = ('Hue', 'Saturation', 'Value')
cs_ycrcb = ('Y (Luma)', 'Cr', 'Cb')
cs_lab = ('Lightness', 'a', 'b')
cs_grey_scale = ['Grey']


class ColorDensityCube:
    """
    Tool for 3D representation of image color distribution over the whole color spectrum.
    It was made to keep the relation between the 3 color channels
    """
    def __init__(self, resolution=16, cube=None):
        assert resolution and not resolution & (resolution - 1)  # resolution is a power of 2
        self.res = resolution
        if not cube:
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
        """
        Feeds an image in the cube, meaning the image's colors distribution will be projected onto the cube
        :param image:
        """
        for x in image:
            for y in x:
                c0 = int(y[0]/self.win)
                c1 = int(y[1]/self.win)
                c2 = int(y[2]/self.win)
                self.cube[c0][c1][c2] += 1
        self.num += 1
        self.isNormalized = False

    def avg(self):
        """ Averages the cube if multiple images have been fed """
        assert self.num > 0
        return self.cube / self.num

    def normalize(self):
        """Normalize the cube values: values will be between 1 and 0"""
        if self.num != 0:
            self.norm_cube = np.array(self.avg())
        else:
            self.norm_cube = np.array(self.cube)
        vmax, vmin = abs(self.norm_cube).max(), 0  # abs(self.norm_cube).min()
        if not vmax:
            print('Cube is null')
            print(self.norm_cube)
        self.norm_cube = (self.norm_cube - vmin) / (vmax - vmin)
        self.isNormalized = True

    def substract(self, cube, state='avg'):
        """
        Subtracts cube to self
        :param cube: target cube to subtract to self
        :param state:
        :return:
        """
        diff_cube = ColorDensityCube(self.res)
        assert isinstance(cube, ColorDensityCube)
        assert state in {'avg', 'norm', 'value'}
        for x in xrange(len(self.cube)):
            if state == 'avg':
                assert self.get_num() and cube.get_num()
                num = np.subtract(self.avg()[x], cube.avg()[x])
            elif state == 'norm':
                num = np.subtract(self.get_normalized()[x], cube.get_normalized()[x])
            else:  # state == 'value':
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
        plt.show()
        if save:
            assert title is not None
            fig.text(0.5, 0.975, title, ha='center')
            plt.savefig(path + title + '.png')
        plt.close()


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


def color_domains_accuracy(model, granularity=4, n=1, data_range=(50000, 60000)):
    g = granularity
    images_cube = dt.cifar10_nth_maxcolor_domains(granularity=g, n=n, data_range=data_range)
    scores_cube = np.zeros((g, g, g))
    data = dt.get_data('cifar10', data_range)
    xf, yf = dt.format_data(data, 10)
    for x in xrange(g):
        for y in xrange(g):
            for z in xrange(g):
                test_data = [[], []]
                if len(images_cube[x][y][z]) > 1:
                    for k in images_cube[x][y][z]:
                        test_data[0].append(xf[k])
                        test_data[1].append(yf[k])
                    # print(np.array(test_data[0]).shape)
                    y_predicted = model.predict(np.array(test_data[0]))
                    predicted_classes = np.argmax(y_predicted, axis=1)
                    true_classes = np.argmax(test_data[1], axis=1)
                    acc = metrics.accuracy(predicted_classes, true_classes)
                else:
                    acc = None
                scores_cube[x][y][z] = acc
    return scores_cube


def colorfulness(image):
    """
    Computes a colorfulness measure
    From: https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    :param image:
    :return:
    """

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
    stdroot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanroot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdroot + (0.3 * meanroot)


def contrast(img):
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    m = np.mean(img_ycc[0])
    vmin, vmax = np.min(img_ycc[0]), np.max(img_ycc[0])
    return m/(vmax-vmin)


# ====== # Finetuning # ====== #


def finetune_by_region(region_coord, ft_data_src, n_data, resolution):
    """
    return the n images which colors of region_coord are the most present. ascending order.
    :param region_coord:
    :param ft_data_src:
    :param n_data:
    :param resolution:
    :return:
    """
    region_scores = []  # same order as ft_data_src
    for image in ft_data_src[0]:
        c = ColorDensityCube(resolution=resolution)
        c.feed(image)
        region_scores.append(c.get_cube()[region_coord[0]][region_coord[1]][region_coord[2]])
    sorted_args = np.argsort(region_scores)
    return sorted_args[-n_data:]


def finetune_by_cdc(high_pr, low_pr, test_data_orig, ft_data_src, cube_res=8, plot_cdcs=False,
                    plot_cdcs_save=True, model_name='unknown_model', res_path=''):
    """
    Process of selection for fine-tuning experiments. Uses the ColorDensityCube class to identify which data is lacking.
    :param high_pr: List[int] List of ids of images with high-rated predictions
    :param low_pr:  List[int] List of ids of images with low-rated predictions
    :param test_data_orig: Source of data analyse model weaknesses.
    :param cube_res: ColorDensityCube resolution for image analysis
    :param ft_data_src: Source of data for selection, should correspond to ids of high/low_pr.
    :param plot_cdcs: bool To plot or not the CDCs used for the process
    :param plot_cdcs_save: to save or not the CDCs if plotted
    :param model_name: Model's name for plots
    :param res_path: Path for saving plots
    :return: List[int] Indexes of the selected data for the fine-tuning experiment.
    """

    # study test data color distrib
    cdc_high = ColorDensityCube(resolution=cube_res)
    for index in high_pr:
        cdc_high.feed(test_data_orig[0][index])
    cdc_high.normalize()
    if plot_cdcs:
        cdc_high.plot_cube(save=plot_cdcs_save, title=model_name + '-high_pr', path=res_path)

    cdc_low = ColorDensityCube(resolution=cube_res)
    for index in low_pr:
        cdc_low.feed(test_data_orig[0][index])
    cdc_low.normalize()
    if plot_cdcs:
        cdc_low.plot_cube(save=plot_cdcs_save, title=model_name + '-low_pr', path=res_path)

    cdc_diff = cdc_high.substract(cdc_low, state='norm')  # What data does the model predicts better?
    # cdc_diff.plot_cube()

    # Fine-tune data selection
    cdc_finetune = ColorDensityCube(resolution=cube_res)
    finetune_data_args = get_best_scores(ft_data_src[0], 10000, cdc_diff)

    for img_index in finetune_data_args:
        cdc_finetune.feed(ft_data_src[0][img_index])
    cdc_finetune.normalize()
    if plot_cdcs:
        cdc_finetune.plot_cube(save=plot_cdcs_save, title=model_name + '-ft_selection', path=res_path)

    return finetune_data_args


def finetune_by_colorfulness(ft_data_src, num):
    col_scores =[]
    for img in ft_data_src:
        col_scores.append(colorfulness(img))
    col_scores = np.argsort(col_scores)
    return col_scores[:num]
