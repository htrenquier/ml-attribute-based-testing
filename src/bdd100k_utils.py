from __future__ import division
import json
from datetime import datetime
import os
import data_tools as dt


def annotate(input_json, output_csv, data_path, overwrite=False):
    """
    Annotate for RetinaNet https://github.com/fizyr/keras-retinanet
    Only annotates the object classes (not areas for segmentation)
    :param input_json: json input file path
    :param output_csv: csv annotation file path
    :param data_path: path of data
    :param overwrite:
    :return:
    """
    fd_json = open(input_json, 'r')
    y = json.load(fd_json)
    fd_json.close()
    start_time = datetime.now()
    if os.path.isfile(output_csv) and not overwrite:
        print('File ' + output_csv + ' already exists. Not written.')
        return
    fd_out = open(output_csv, 'w')
    object_classes = []
    for img_id in xrange(len(y)):
        name = y[img_id][u'name']
        is_empty = True
        for label in y[img_id][u'labels']:
            cat = label[u'category']
            if cat not in object_classes:
                if u'box2d' in label.keys():
                    object_classes.append(cat)
                else:
                    continue
            b2d = label[u'box2d']
            x1, y1, x2, y2 = b2d['x1'], b2d['y1'], b2d['x2'], b2d['y2']
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            if int(x_max) <= int(x_min) or int(y_max) <= int(y_min):
                continue
            row = ('%s,%d,%d,%d,%d,%s\n' % (data_path + name, x_min, y_min, x_max, y_max, cat))
            fd_out.write(row)
            is_empty = False
        if is_empty:
            row = ('%s,,,,,\n' % (data_path + name))
            fd_out.write(row)
    fd_out.close()
    print('File successfully written', output_csv, 'in (s)', str(datetime.now() - start_time))
    return object_classes


def annotate_tiny(input_json, output_path, data_path, overwrite=False):
    """
    Annotate a subset (10k) of the bdd100k from the validation data
    :param input_json:
    :param output_path:
    :param data_path:
    :param overwrite:
    :return:
    """
    if os.path.isfile(output_path + 'tiny_train_annot.csv') and not overwrite:
        print('File ' + output_path + 'tiny_train_annot.csv' + ' already exists. Not written.')
    else:
        annotate_tiny_range(input_json, output_path+'tiny_train_annot.csv', data_path, xrange(0, 70), overwrite)

    if os.path.isfile(output_path + 'tiny_val_annot.csv') and not overwrite:
        print('File ' + output_path + 'tiny_val_annot.csv' + ' already exists. Not written.')
    else:
        annotate_tiny_range(input_json, output_path+'tiny_val_annot.csv', data_path, xrange(70, 80), overwrite)

    return output_path+'tiny_train_annot.csv', output_path+'tiny_val_annot.csv'


def annotate_tiny_range(input_json, output_csv, data_path, range, overwrite=False):
    fd_json = open(input_json, 'r')
    y = json.load(fd_json)
    fd_json.close()
    if os.path.isfile(output_csv) and not overwrite:
        print('File ' + output_csv + ' already exists. Not written.')
        return
    fd_out = open(output_csv, 'w')
    object_classes = []
    for img_id in range:
        name = y[img_id][u'name']
        is_empty = True
        for label in y[img_id][u'labels']:
            cat = label[u'category']
            if cat not in object_classes:
                if u'box2d' in label.keys():
                    object_classes.append(cat)
                else:
                    continue
            b2d = label[u'box2d']
            x1, y1, x2, y2 = b2d['x1'], b2d['y1'], b2d['x2'], b2d['y2']
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            if int(x_max) <= int(x_min) or int(y_max) <= int(y_min):
                continue
            row = ('%s,%d,%d,%d,%d,%s\n' % (data_path + name, x_min, y_min, x_max, y_max, cat))
            fd_out.write(row)
            is_empty = False
        if is_empty:
            row = ('%s,,,,,\n' % (data_path + name))
            fd_out.write(row)
    fd_out.close()
    return object_classes


def class_mapping(classes=None, input_json=None, output_csv='class_mapping.csv', overwrite=False):
    """
    Writes the class mapping file for objects categories
    :param classes: list of classes or categories names
    :param input_json: file path for json labels
    :param output_csv: file path for class mapping file
    :param overwrite:
    :return: path of output file
    """
    if os.path.isfile(output_csv) and not overwrite:
        print('File ' + output_csv + ' already exists. Not written.')
        return output_csv
    if input_json:
        with open(input_json, 'r') as fd_json:
            y = json.load(fd_json)
        classes = []
        for img_id in xrange(len(y)):
            for label in y[img_id][u'labels']:
                cat = label[u'category']
                if cat not in classes:
                    if u'box2d' in label.keys():
                        classes.append(cat)
                    else:
                        continue
    else:
        if not classes:
            print('No input for class mapping')
            return
    with open(output_csv, 'w') as cl_map:
        for k in xrange(len(classes)):
            cl_map.write(classes[k] + ',' + str(k) + '\n')
        print('Class mapping successfully written')
    return output_csv


def get_label_names(class_map_file):
    labels_to_names = {}
    with open(class_map_file, 'r') as map_file:
        for l in map_file.readlines():
            labels_to_names[int(l.split(',')[1])] = l.split(',')[0]
    print(labels_to_names)
    return labels_to_names


def avg_box_size(annotation_file):
    x = 0
    y = 0
    n = 0
    with open(annotation_file, 'r') as f:
        for l in f.readlines():
            split = l.split(',')
            assert len(split) == 6
            x_min, y_min, x_max, y_max = int(split[1]), int(split[2]), int(split[3]), int(split[4])
            x += (x_max - x_min)
            y += (y_max - y_min)
            n += 1
    x_avg = x / n
    y_avg = y / n
    return x_avg, y_avg, n


def do_boxes_collide(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    if x_max2 <= x_min1:
        return False
    elif x_min2 >= x_max1:
        return False
    else:
        if y_max2 <= y_min1:
            return False
        elif y_min2 >= y_max1:
            return False
        else:
            return True


def get_box(line):
    split = line.split(',')
    return int(split[1]), int(split[2]), int(split[3]), int(split[4])


def annotate_objects(annotation_file, output):
    # Not optimizing the number of boxes
    start_time = datetime.now()
    with open(annotation_file, 'r') as input_annot:
        with open(output, 'w') as out:
            line = input_annot.readline()
            while line:
                curr_id = str(line.split(',')[0])
                boxes = []
                while line and str(line.split(',')[0]) == curr_id:
                    box1 = get_box(line)
                    collision = False
                    for box2 in boxes:
                        if do_boxes_collide(box1, box2):
                            collision = True
                            continue
                    if not collision:
                        boxes.append(box1)
                        out.write(line)
                    line = input_annot.readline()
    print('File successfully written', output, 'in (s)', str(datetime.now() - start_time))


def adjust_ratio(box, ratio):
    x_min, y_min, x_max, y_max = box
    x = x_max - x_min
    y = y_max - y_min
    target_ratio = ratio[0] / ratio[1]
    img_ratio = x / y

    if target_ratio > img_ratio:
        target_width = y * target_ratio
        x_min -= (target_width - x)//2
        x_max += (target_width - x)//2 + (target_width - x) % 2
    elif target_ratio < img_ratio:
        target_height = x / target_ratio
        y_min -= (target_height - y) // 2
        y_max += (target_height - y) // 2 + (target_height - y) % 2
    return x_min, y_min, x_max, y_max


def adjust_size(box, format):
    # Adds margin if formatted image is too small
    x_min, y_min, x_max, y_max = box
    if (x_max - x_min) < format[0]:
        x = x_max - x_min
        y = y_max - y_min
        x_min -= (format[0] - x) // 2
        x_max += (format[0] - x) // 2 + (format[0] - x) % 2
        y_min -= (format[1] - y) // 2
        y_max += (format[1] - y) // 2 + (format[1] - y) % 2
        assert (x_max - x_min) == format[0] and (y_max - y_min) == format[1]
    return x_min, y_min, x_max, y_max


def adjust_position(box, image_size):
    x_min, y_min, x_max, y_max = box
    x = x_max - x_min
    y = y_max - y_min
    if x > image_size[0] or y > image_size[1]:
        return None
    if x_min < 0:
        x_max -= x_min
        x_min -= x_min
    elif x_max > image_size[0]:
        x_min -= x_max - image_size[0]
        x_max -= x_max - image_size[0]
    elif y_min < 0:
        y_max -= y_min
        y_min -= y_min
    elif y_max > image_size[1]:
        y_min -= y_max - image_size[1]
        y_max -= y_max - image_size[1]
    return x_min, y_min, x_max, y_max


def build_dataset(obj_annot_file, output_path, labels_file):
    """
    Builds classification dataset from object detection dataset.
    For BDD100k: 1044675 training images from training folder
                150738 val images from val folder
    Not optimized to maximise number of images.
    Not time-optimized
    Minimum size of object: 44 pixels horizontally or vertically from original image
    images
    :param obj_annot_file: annotation file of format https://github.com/fizyr/keras-retinanet
    :param output_path: where images will be saved
    :param labels_file: names of files for classification ground truth
    :return:
    """
    min_size = 44  # set image size = 64x64, max margin = 20
    format = (64, 64)
    img_size = (1280, 720)
    cnt = 0
    print('Starting extraction...')
    start_time = datetime.now()
    obj_annot = open(obj_annot_file, 'r')
    labels_fd = open(labels_file, 'w')
    line = obj_annot.readline()
    curr = line.split(',')[0]
    boxes = []
    classes = []

    while line:
        if line.split(',')[0] != curr:
            _, file_names = dt.crop_resize(curr, boxes, resize_format=format, output_path=output_path)
            for i in xrange(len(classes)):
                labels_fd.write(file_names[i] + ',' + classes[i])
            # next image
            boxes = []
            classes = []
            curr = line.split(',')[0]

        box = get_box(line)
        if box[2] - box[1] < min_size:
            line = obj_annot.readline()
            continue
        else:
            box = adjust_ratio(box, format)
            box = adjust_size(box, format)
            box = adjust_position(box, img_size)
            if not box:
                line = obj_annot.readline()
                continue
            boxes.append(box)
            classes.append(line.split(',')[-1])
            cnt += 1
        line = obj_annot.readline()
    _, file_names = dt.crop_resize(curr, boxes, resize_format=format, output_path=output_path)
    assert len(classes) == len(file_names)
    for i in xrange(len(classes)):
        labels_fd.writelines('%s,%s' % (file_names[i], classes[i]))

    obj_annot.close()
    labels_fd.close()

    print(str(cnt) + ' images successfully generated in ' + output_path + ' in '
          + str(datetime.now() - start_time) + '(s)')


def get_ids_labels(labels_file, class_map_file):
    name_to_label = {}
    with open(class_map_file, 'r') as map_file:
        for l in map_file.readlines():
            name_to_label[str(l.split(',')[0])] = int(l.split(',')[1])

    id_list = []
    labels = []
    with open(labels_file, 'r') as gt_fd: # ground_truth_fd
        line = gt_fd.readline()
        while line:
            s = line.split(',')
            id_list.append(s[0])
            labels.append(name_to_label[s[0].rstrip()])
            line = gt_fd.readline()
    return id_list, labels