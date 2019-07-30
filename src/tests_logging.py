from os import path
import metrics as m
import numpy as np


class Log:
    def __init__(self, log_filename, **col_names):
        self.log_filename = log_filename + '.csv'
        self.nb_col = len(col_names)
        self.vals = []
        if col_names:
            with open(self.log_filename, 'w') as f:
                f.write(",".join(tuple(col_names)))

    def w_line(self, line):
        with open(self.log_filename, 'w+') as f:
            f.write(str(line))

    def w_array(self, array):
        with open(self.log_filename, 'w+') as f:
            f.write(str(array)[1:-1])

    def add_val(self, val):
        self.vals.append(val)

    def w_vals(self):
        self.w_array(self.vals)
        self.vals = []


def read_ground_truth(gt_file):
    """
    Ground Truth read function for imagenet validation data
    :param gt_file: file available @ http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    :return: List[str], List[int] with ground truth file name and class
    """
    true_classes = []
    filenames = []
    with open(gt_file, 'r') as f:
        for l in f.readlines():
            line = l.split()
            filenames.append(line[0].strip())
            true_classes.append(int(line[1].strip()))
    return filenames, true_classes


def log_predictions(y_predicted, model_name, file_path, tag=''):
    model_file = model_name.rstrip('.h5') + '_' + tag + '-predictions.csv'
    if path.isfile(file_path + model_file):
        print('File ' + file_path + model_file + ' already exists. Not written.')
        return
    f = open(file_path + model_file, "w+")
    for pred in y_predicted:
        f.write(str(pred)[1:-1])
    f.close()
    print(file_path + model_file + ' written.')


def load_predictions(model_name, file_path, tag=''):
    model_file = model_name.rstrip('.h5') + '_' + tag + '-predictions.csv'
    f = open(file_path + model_file, "r")
    y_predicted = []
    for l in f.readlines():
        y_predicted.append([float(k) for k in l.split(' ')])
    return y_predicted




def log_predictions2(y_predicted, model_name, file_path):
    model_file = model_name + '-res.csv'
    if not path.isfile(file_path + model_file):
        f = open(file_path + model_file, "w+")
        for i in xrange(len(y_predicted)):
            line = '{0}, {1}, {2}, {3}\r\n'.format(str(i),
                                                   str(m.confidence(y_predicted[i])),
                                                   str(np.argmax(y_predicted[i])),
                                                   str(y_predicted[i]))
            f.write(line)
        f.close()
        # print('Predictions for ' + model_file + ' written.')
    # else:
        # print('Predictions for ' + model_file + ' already written!')


def load_csv(file_name, col):
    """
    Extracts info from csv file from test_bench
    :param file_name: file to get info from
    :param col: column in the csv file to get info from
            0: image id
            1: confidence
            2: predicted class
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
                    info.append(int(infos[col]))

            str = ''
    f.close()
    return info
