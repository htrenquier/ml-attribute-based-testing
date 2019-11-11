from __future__ import division
import numpy as np
from sklearn import metrics as sk_metrics
import metrics
import metrics_color
import plotting
import model_trainer as mt
import data_tools as dt
import tests_logging as t_log
import initialise
from itertools import product

# Paths
csv_path = '../res/csv/'
png_path = '../res/png/'
h5_path = '../res/h5/'

# models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50')
models = ['densenet121', 'resnet50']


class MetricStructure:
    def __init__(self, entries_lists):
        self.entries_lists = list(entries_lists)
        self.struct = {tuple(key): [] for key in product(*entries_lists)}
        self.dims = [len(el) for el in entries_lists]
        print(self.struct)

    def set_value(self, entries_ids, value):
        self.struct[tuple(entries_ids)].append(value)

    def get_value(self, entries_ids):
        return np.mean(self.struct[entries_ids])

    def get_means(self):
        means = np.zeros(self.dims)
        for index in product(*[xrange(k) for k in self.dims]):
            key = [self.entries_lists[c][entry] for c, entry in enumerate(list(index))]
            means[index] = np.mean(self.struct[tuple(key)])
        return means

    def get_table_rec(self, entries_lists, arr):
        if not entries_lists:
            return []
        else:
            return [self.get_table_rec(entries_lists[:-1], arr[k]) for k in xrange(len(entries_lists))]

    def get_entries_lists(self):
        return self.entries_lists


class DiscreteAttribute:
    def __init__(self, key_to_value):
        self.valueof = key_to_value
        self.uniques = np.asarray(np.unique(key_to_value.values(), return_counts=True))
        self.indexof = {self.uniques[0][k]: k for k in xrange(len(self.uniques[0]))}
        self.metrics = dict()

    def __getitem__(self, id):
        return self.valueof[id]

    def get_distribution(self):
        return self.uniques

    def set_value(self, metric_name, data_metric_key, entries_ids, value):
        entries_ids.append(self.valueof[data_metric_key])
        self.metrics[metric_name].set_value(entries_ids, value)

    def add_metric(self, name, entries_lists):
        # List of attributes excluding current attribute
        entries_lists.append(self.uniques[0].tolist())
        metric = MetricStructure(entries_lists)
        self.metrics.update({name: metric})

    def get_metric_value(self, metric_name, data_metric_key, entries_ids):
        entries_ids.append(self.valueof[data_metric_key])
        return self.metrics[metric_name].get_value(entries_ids)

    def get_metric_means(self, name):
        # 2D metrics printing
        return [",".join([str(f) for f in row]) + '\n' for row in self.metrics[name].get_means()]



def colorcube_analysis():
    # m = 'densenet121'
    for m in models:
        test_data = dt.get_data('cifar10', (50000, 60000))
        top_n = 2500
        # model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
        model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False, suffix='ft20ep-exp')
        model = mt.load_by_name(model_name0, test_data[0].shape[1:], h5_path+model_name0)
        y_predicted = model.predict(np.array(test_data[0]))
        # y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
        true_classes = [int(k) for k in test_data[1]]
        scores = metrics.prediction_ratings(y_predicted, true_classes)
        score_sorted_ids = np.argsort(scores)
        cc_high = metrics_color.ColorDensityCube(resolution=4)
        for img_id in score_sorted_ids[-top_n:]:
            cc_high.feed(test_data[0][img_id])
        cc_high.normalize()
        cc_high.plot_cube()

        cc_low = metrics_color.ColorDensityCube(resolution=4)
        for img_id in score_sorted_ids[:top_n]:
            cc_low.feed(test_data[0][img_id])
        cc_low.normalize()

        cc_diff = cc_high.substract(cc_low, 'norm')

        cc_low.plot_cube()

        # cc_diff.normalize()
        cc_diff.plot_cube(title='Color cube analysis difference (' + str(top_n) + ' images/series)', normalize=False,
                          save=True)


def histogram_analysis():
    m = 'densenet121'
    test_data = dt.get_data('cifar10', (50000, 60000))
    top_n = 1000
    model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    true_classes = [int(k) for k in test_data[1]]
    scores = metrics.prediction_ratings(y_predicted, true_classes)
    score_sorted_ids = np.argsort(scores)
    high_score_series = []
    low_score_series = []
    for k in xrange(0, top_n):
        high_score_series.append(test_data[0][score_sorted_ids[-k-1]])
        low_score_series.append(test_data[0][score_sorted_ids[k]])

    plotting.plot_hists(high_score_series, 'high scores', low_score_series, 'low scores', plotting.cs_bgr,
                        title='Histogram analysis (' + str(top_n) + ' images/series)')


def colorfulness_analysis(model='densenet121', top_n=2500):
    """
    Experiment to analyse the relevance if the colorfulness attribute
    See the metrics_color.colorfulness() function for more details on the attribute
    :param model: The predictions of :model: will be used to compute the prediciton scores
    :param top_n: Number of elements in the series that will be plotted for analysis
    :return:
    """

    # Load test data and model results
    test_data = dt.get_data('cifar10', (50000, 60000))
    model_name0 = mt.weight_file_name(model, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    true_classes = [int(k) for k in test_data[1]]

    # Compute scores and sort test data ids by score
    scores = metrics.prediction_ratings(y_predicted, true_classes)
    score_sorted_ids = np.argsort(scores)

    # Compute metric for high score and low score data
    high_score_series = []
    low_score_series = []
    print(len(score_sorted_ids))
    for k in xrange(0, top_n):
        high_score_series.append(metrics_color.colorfulness(test_data[0][score_sorted_ids[-k-1]]))
        low_score_series.append(metrics_color.colorfulness(test_data[0][score_sorted_ids[k]]))

    # Plot box plot of the two series
    plotting.box_plot(high_score_series, low_score_series, name_s1='high prediction scores',
                      name_s2='low prediction scores', y_label='Colorfulness',
                      title='Colorfulness analysis (' + str(top_n) + ' images/series)')


def entropy_cc_analysis():
    m = 'densenet121'
    test_data = dt.get_data('cifar10', (50000, 60000))
    top_n = 2500

    model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    true_classes = [int(k) for k in test_data[1]]
    scores = metrics.prediction_ratings(y_predicted, true_classes)
    score_sorted_ids = np.argsort(scores)
    high_score_entropies = []
    low_score_entropies = []
    print(len(score_sorted_ids))
    for k in xrange(0, top_n):
        # id = score_sorted_ids[-k - 1]
        # print(id)
        # img = test_data[id]
        high_score_entropies.append(metrics_color.entropy_cc(test_data[0][score_sorted_ids[-k-1]], 8))
        low_score_entropies.append(metrics_color.entropy_cc(test_data[0][score_sorted_ids[k]], 8))

    plotting.box_plot(high_score_entropies, low_score_entropies, name_s1='high prediction scores',
                      name_s2='low prediction scores',y_label='Color cube entropy',
                      title='Entropy analysis (' + str(top_n) + ' images/series)')


def data_analysis():

    tr_data = dt.get_data('cifar10', (0, 20000))
    val_data = dt.get_data('cifar10', (40000, 50000))
    test_data = dt.get_data('cifar10', (50000, 60000))

    for m in models:
        # model0, model_name0 = mt.train2(m, tr_data, val_data, 50, False, 'cifar10-2-5', h5_path)
        # model0, model_name0 = mt.train(m, 'cifar10-channelswitched', 50, data_augmentation=False, path=res_path)
        # acc, predicted_classes, y_predicted = dt.predict_and_acc(model0, test_data)
        # t_log.log_predictions(y_predicted, model_name0, file_path=csv_path)

        model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
        y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)

        # true_classes = np.argmax(test_data[1], axis=1)  # wrong
        true_classes = [int(k) for k in test_data[1]]
        pr = metrics.prediction_ratings(y_predicted, true_classes)
        scores = []

        for image in test_data[0]:
            scores.append(metrics_color.colorfulness(image))

        max = np.max(scores)
        index = list(scores).index(max)
        scores.pop(index)
        pr.pop(index)

        plotting.quick_plot(pr, scores, png_path+model_name0+'contrast.png')


def pr_on_fair_distribution(models=['densenet121'], top_n=100, res=4):
    test_data = dt.get_data('cifar10', (50000, 60000))

    # Add every image's cube in densities
    densities = []
    for img in test_data[0]:
        cc = metrics_color.ColorDensityCube(res)
        cc.feed(img)
        densities.append(cc.get_cube())
        # ccf = np.array(cc.get_cube()).flatten()

    # Shape densities (list of cubes) to make a list per color
    densities_lists = np.swapaxes(np.swapaxes(np.swapaxes(densities, 0, 3), 0, 2), 0, 1)
    # print(densities_lists.shape)
    densities_cube = np.empty((res, res, res), dtype=object)

    # For each color keep the ids of the top_n most dense images in this color (same image can be in 2 colors)
    for i in xrange(res):
        for j in xrange(res):
            for k in xrange(res):
                # pr_most_dense = []
                density_list = densities_lists[i][j][k].tolist()
                args_most_dense = np.argsort(density_list)[-top_n:]
                densities_cube[i][j][k] = args_most_dense
    # print(densities_cube.shape)

    # Per model analysis
    for m in models:
        # Load model predictions and ground_truth values
        model_name0 = mt.weight_file_name(m, 'cifar10-2-5', 50, False)
        y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
        true_classes = [int(k) for k in test_data[1]]
        pr = metrics.prediction_ratings(y_predicted, true_classes)

        # For each color get prediction score of the top_n images
        score_cube = np.zeros((res, res, res))
        global_cc = metrics_color.ColorDensityCube(resolution=res)
        args_most_dense_all = []
        for i in xrange(res):
            for j in xrange(res):
                for k in xrange(res):
                    pr_most_dense = []
                    densities_args = densities_cube[i][j][k].tolist()
                    # args_most_dense = np.argsort(density_list)[-topn:]
                    ijk_cc = metrics_color.ColorDensityCube(res)
                    for a in densities_cube[i][j][k].tolist():
                        pr_most_dense.append(pr[a])
                        ijk_cc.feed(test_data[0][a])
                        global_cc.feed(test_data[0][a])
                    ijk_cc.normalize()
                    ttl = 'color = (' + str(float(i/res)) + ', ' + str(float(j/res)) + ', ' + str(float(k/res)) + ')'
                    # ijk_cc.plot_cube()
                    score_cube[i][j][k] = np.mean(pr_most_dense)
                    print(np.mean(pr_most_dense))
                    # args_most_dense_all.append(args_most_dense)
                    ttl = 'color = (' + str(float(i/res)) + ', ' + str(float(j/res)) + ', ' + str(float(k/res)) + ')'
                    # plotting.show_imgs(densities_args[:10], ttl, test_data[0], showColorCube=True, resolution=4)

        global_cc.normalize()
        global_cc.plot_cube(title='Fair distributed dataset ColorCube')

        sc = metrics_color.ColorDensityCube(resolution=res, cube=score_cube)
        sc.normalize()
        sc.plot_cube(title='Scores per color for ' + m)


def confusion(model='densenet121'):
    # Load test data and model results
    test_data = dt.get_data('cifar10', (50000, 60000))
    model_name0 = mt.weight_file_name(model, 'cifar10-2-5', 50, False)
    y_predicted = t_log.load_predictions(model_name0, file_path=csv_path)
    predicted_classes = np.argmax(y_predicted, axis=1)
    true_classes = [int(k) for k in test_data[1]]

    print('Confusion Matrix for Total Test Data')
    print(sk_metrics.confusion_matrix(true_classes, predicted_classes))

    scores = metrics.prediction_ratings(y_predicted, true_classes)
    prediction_scores = np.zeros((10, 1)).tolist()
    print(prediction_scores)
    for k in xrange(len(y_predicted)):
        prediction_scores[predicted_classes[k]].append(scores[k])

    print(np.array(prediction_scores).shape)
    for cifar_class in prediction_scores:
        print(float(np.mean(cifar_class)))


def main():
    """
    Attribute analysis
    :return:
    """
    # colorcube_analysis()
    # histogram_analysis()
    # entropy_cc_analysis()
    # colorfulness_analysis()
    # r_on_fair_distribution()
    # data_analysis()
    # confusion()

main()