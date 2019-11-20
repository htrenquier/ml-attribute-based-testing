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
import bdd100k_utils as bu
import cv2
import os.path
import json
from itertools import product

# Paths
csv_path = '../res/csv/'
png_path = '../res/png/'
h5_path = '../res/h5/'
labels_path = '../../bdd100k/classification/labels/'
bdd100k_labels_path = "../../bdd100k/labels/"
box_tr_file = '../../bdd100k/classification/labels/train_ground_truth_attributes.csv'
val_labels_csv = '../../bdd100k/classification/labels/val_ground_truth.csv'
box_val_file = '../../bdd100k/classification/labels/val_ground_truth_attributes.csv'
val_json = '../../bdd100k/labels/bdd100k_labels_images_val.json'
attr_tr_file = bdd100k_labels_path + 'bdd100k_labels_images_train_attributes.csv'
attr_val_file = bdd100k_labels_path + 'bdd100k_labels_images_val_attributes.csv'
box_tr_json = labels_path + 'box_size_train_attribute.json'
class_map_file = labels_path + 'class_mapping.csv'
box_val_json = labels_path + 'box_size_val_attribute.json'
labels_path = '../../bdd100k/classification/labels/'

# models = ('densenet121', 'mobilenet', 'mobilenetv2', 'nasnet', 'resnet50')
models = ['densenet121', 'resnet50']


class MultidimMetricStructure:
    def __init__(self, entries_lists):
        self.entries_lists = list(entries_lists)
        self.struct = {tuple(key): [] for key in product(*entries_lists)}
        self.dims = [len(el) for el in entries_lists]

    def flush(self):
        self.struct = {tuple(key): [] for key in product(*self.entries_lists)}

    def set_value(self, entries_ids, value):
        self.struct[tuple(entries_ids)].append(value)

    def get_value_list(self, entries_ids):
        return self.struct[tuple(entries_ids)]

    def get_value_mean(self, entries_ids):
        return np.mean(self.struct[tuple(entries_ids)])

    def global_mean(self):
        global_arr = []
        for key in self.struct.keys():
            global_arr += self.struct[key]
        return np.mean(global_arr)

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


class MetricStructure:
    def __init__(self, entries):
        self.entries = list(entries)
        self.struct = {entry: [] for entry in self.entries}

    def flush(self):
        self.struct = {entry: [] for entry in self.entries}

    def add_value(self, entry, value):
        self.struct[entry].append(value)

    def get_value_list(self, entry):
        return self.struct[entry]

    def get_value_mean(self, entry):
        return np.mean(self.struct[entry])

    def global_mean(self):
        global_arr = []
        for key in self.struct.keys():
            global_arr += self.struct[key]
        return np.mean(global_arr)

    def get_means(self):
        means = np.zeros(len(self.entries))
        for i, entry in enumerate(self.entries):
            means[i] = np.mean(self.struct[entry])
        return means


class DiscreteAttribute:
    """
    Stores data ID to attribute's labels dictionary and later stores values in MetricStructure for different metrics
    """

    def __init__(self, key_to_label, metric_name='score'):
        self.metrics = dict()
        self.key_to_label = key_to_label
        self.uniques = np.asarray(np.unique(key_to_label.values(), return_counts=True))
        self.metrics.update({metric_name: MetricStructure(self.uniques[0])})
        self.indexof = {self.uniques[0][k]: k for k in xrange(len(self.uniques[0]))}

    def __getitem__(self, id):
        return self.uniques[0][id]

    def flush(self):
        for m in metrics.values():
            m.flush()

    def get_labels(self):
        return [str(l) for l in self.uniques[0]]

    def labelof(self, key):
        return self.key_to_label[key]

    def index_of(self, label):
        return self.indexof[label]

    def get_distribution(self):
        return [str(c) for c in self.uniques[1]]

    def add_value(self, metric_name, value, data_key):
        self.metrics[metric_name].add_value(self.key_to_label[data_key], value)

    def add_metric(self, name):
        metric = MetricStructure(self.uniques[0])
        self.metrics.update({name: metric})

    def get_metric_value_list(self, metric_name, label):
        """

        :param metric_name:
        :param label: Attribute's label
        :return:
        """
        return self.metrics[metric_name].get_value_list(label)

    def get_metric_mean(self, metric_name, label):
        return self.metrics[metric_name].get_value_mean(label)

    def get_metric_means(self, metric_name):
        return [m for m in self.metrics[metric_name].get_means()]

    def log_headers(self, fd):
        fd.write(",".join(self.get_labels()) + '\n')
        fd.write(",".join(self.get_distribution()) + '\n')

    def log_metric_means(self, name, fd):
        fd.writelines(",".join([str(m) for m in self.get_metric_means(name)]) + '\n')

    def get_metric_global_mean(self, name):
        return self.metrics[name].global_mean()


def bdd100k_discrete_attribute_analyse(model_file, attribute, data_ids, y_scores, img_ids):
    res_file = csv_path + model_file.split('_')[0] + '_' + attribute['name'] + '_res_metrics.csv'

    for metric in attribute['metrics']:
        for i in data_ids:
            attribute['d_attribute'].add_value(metric, y_scores[i], attribute['dk2ak'](img_ids[i]))

    fd = open(res_file, 'w')
    for metric in attribute['metrics']:
        attribute['d_attribute'].log_headers(fd)
        attribute['d_attribute'].log_metric_means(metric, fd)
        fd.write(str(attribute['d_attribute'].get_metric_global_mean(metric)))
    fd.close()


def bdd100k_model_analysis(model_file, attributes, val_labels):
    print("")
    print(" =#= Analysing " + model_file.split('_')[0] + " =#= ")
    print("")

    threshold = 0
    pr_file = '.'.join(model_file.split('.')[:-1]) + '_predictions.csv'
    predictions, y_scores, img_ids = dt.get_scores_from_file(csv_path + pr_file, val_labels)
    # 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
    for n_data in [100000]:
        # test_subset creation
        top_n_args, bot_n_args = dt.get_topbot_n_args(n_data, y_scores)
        for attribute in attributes.values():
            # Attribute init + analysis
            attribute['d_attribute'] = DiscreteAttribute(attribute['map'])
            for metric in attribute['metrics']:
                attribute['d_attribute'].add_metric(metric)
            bdd100k_discrete_attribute_analyse(model_file, attribute, bot_n_args, y_scores, img_ids)
            for mc, metric in enumerate(attribute['metrics']):
                for lc, label_mean in enumerate(attribute['d_attribute'].get_metric_means(metric)):
                    # print(c, label_mean, attr_mean - threshold)
                    metric_mean = attribute['d_attribute'].get_metric_global_mean(metric)
                    if label_mean < metric_mean - threshold:
                        attribute['weaks'][mc].append(attribute['d_attribute'].get_labels()[lc])

            print(attribute['weaks'])


def select_ft_data(model_file, ft_partition, ft_attribute=None, ft_label=None, do_plot_boxes=False):
    assert (bool(ft_attribute) == bool(ft_label))
    attributes = bdd100k_analysis(model_file, do_plot_boxes)

    attributes['weather']['map'], \
    attributes['scene']['map'], \
    attributes['timeofday']['map'], wst_dk2ak = bu.wst_attribute_mapping(attr_tr_file)
    attributes['box_size']['map'], box_size_dk2ak = bu.box_size_attribute_mapping(box_tr_file, box_tr_json)

    if ft_label and ft_attribute:
        print('Selecting data for ' + ft_attribute + ' / ' + ft_label)
        sel_partition = local_ft_selection(attributes[ft_attribute], ft_label, ft_partition)
    else:
        print('Selecting data for global fine-tuning.')
        sel_partition = global_ft_selection(attributes, ft_partition)
    return sel_partition


def bdd100k_analysis(model_file, do_plot_boxes=False):
    class_map_file = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    # Dataset for analysis
    val_partition, val_labels = bu.get_ids_labels(val_labels_csv, class_map_file)

    # Attribute mapping and data_key to attr_key function (dk2ak)
    weather, scene, timeofday, wst_dk2ak = bu.wst_attribute_mapping(attr_val_file)
    box_size, box_size_dk2ak = bu.box_size_attribute_mapping(box_val_file, box_val_json)

    attributes = {'weather': {'name': 'weather',
                              'map': weather,
                              'dk2ak': wst_dk2ak,
                              'd_attribute': None,
                              'metrics': ['score'],
                              'weaks': [[]]},
                  'scene': {'name': 'scene',
                            'map': scene,
                            'dk2ak': wst_dk2ak,
                            'd_attribute': None,
                            'metrics': ['score'],
                            'weaks': [[]]},
                  'timeofday': {'name': 'timeofday',
                                'map': timeofday,
                                'dk2ak': wst_dk2ak,
                                'd_attribute': None,
                                'metrics': ['score'],
                                'weaks': [[]]},
                  'box_size': {'name': 'box_size',
                               'map': box_size,
                               'dk2ak': box_size_dk2ak,
                               'd_attribute': None,
                               'metrics': ['score'],
                               'weaks': [[]]},
                  }

    bdd100k_model_analysis(model_file, attributes, val_labels)

    # Score for day and night
    day_score = attributes['timeofday']['d_attribute'].get_metric_mean('score', 'daytime')
    night_score = attributes['timeofday']['d_attribute'].get_metric_mean('score', 'night')
    print('Scores: day:', day_score, ' / night:', night_score)

    if do_plot_boxes:
        plotting.plot_discrete_attribute_scores(attributes, model_file.split('_')[0])

    return attributes


def local_ft_selection(attribute, label, ft_partition):
    sel_partition = []
    count = 0
    for data_key in ft_partition:
        if attribute['map'][attribute['dk2ak'](data_key)] == label:
            count += 1
            sel_partition.append(data_key)
    print(str(count) + " data selected.")
    return sel_partition


def global_ft_selection(attributes, ft_partition, n_sel_data):
    sel_partition_occurences = [[] for _ in xrange(len(attributes) + 1)]

    for data_key in ft_partition:
        count = 0
        for attribute in attributes.values():
            if attribute['map'][attribute['dk2ak'](data_key)] in attribute['weaks'][0]:
                count += 1
        sel_partition_occurences[count].append(data_key)

    for k in xrange(len(sel_partition_occurences)):
        print(k, len(sel_partition_occurences[k]))

    sel_partition = []
    k = len(sel_partition_occurences) - 1
    while len(sel_partition) < n_sel_data and k > -1:
        sel_partition = sel_partition + sel_partition_occurences[k]
        print(len(sel_partition))
        k -= 1
    return sel_partition[:n_sel_data]


def bdd100k_cc_analysis():
    model_files = ['densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22.hdf5',
                   ]
                   # 'resnet50_bdd100k_cl0-500k_20ep_woda_ep13_vl0.27.hdf5',
                   # 'mobilenet_bdd100k_cl0-500k_20ep_woda_ep15_vl0.24.hdf5',
                   # 'mobilenetv2_bdd100k_cl0-500k_20ep_woda_ep17_vl0.22.hdf5',
                   # 'nasnet_bdd100k_cl0-500k_20ep_woda_ep17_vl0.24.hdf5']

    class_map_file = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    # Dataset for analysis
    val_partition, val_labels = bu.get_ids_labels(val_labels_csv, class_map_file)

    for m in model_files:

        # test_subset creation
        pr_file = '.'.join(m.split('.')[:-1]) + '_predictions.csv'
        predictions, y_scores, img_ids = dt.get_scores_from_file(csv_path + pr_file, val_labels)
        top_n_args, bot_n_args = dt.get_topbot_n_args(20000, y_scores)

        cc_high = metrics_color.ColorDensityCube(resolution=4)
        for arg in top_n_args:
            cc_high.feed(cv2.imread(img_ids[arg]))
        print('high sum', np.sum(cc_high.get_cube().flatten()))
        cc_high.normalize()
        cc_high.plot_cube()

        cc_low = metrics_color.ColorDensityCube(resolution=4)
        for arg in bot_n_args:
            cc_low.feed(cv2.imread(img_ids[arg]))
        print('low sum', np.sum(cc_low.get_cube().flatten()))
        cc_low.normalize()
        cc_low.plot_cube()

        cc_diff = cc_high.substract(cc_low, 'value')
        print('diff mean', np.sum(cc_diff.get_cube().flatten()))
        print('diff mean', np.mean(cc_diff.get_cube().flatten()))
        cc_diff.normalize()
        # cc_diff.plot_cube()

        # cc_diff.normalize()
        cc_diff.plot_cube(title='Color cube analysis difference (' + str(20000) + ' images/series)', normalize=True,
                          save=True)


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
    return
    # initialise.init()
    # colorcube_analysis()
    # histogram_analysis()
    # entropy_cc_analysis()
    # colorfulness_analysis()
    # r_on_fair_distribution()
    # data_analysis()
    # confusion()
    # select_ft_data('densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22.hdf5', [], 0, do_plot_boxes=True)
    # bdd100k_analysis('densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22_ft_day_ep06_vl0.24.hdf5', do_plot_boxes=True)
    # bdd100k_cc_analysis()

main()