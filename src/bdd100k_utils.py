import json
from datetime import datetime
import os
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.bin import train as kr_train
from keras_retinanet.callbacks import RedirectModel
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau


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
        annotate_tiny_range(input_json, output_path+'tiny_train_annot.csv', data_path, xrange(0, 7000), overwrite)

    if os.path.isfile(output_path + 'tiny_val_annot.csv') and not overwrite:
        print('File ' + output_path + 'tiny_val_annot.csv' + ' already exists. Not written.')
    else:
        annotate_tiny_range(input_json, output_path+'tiny_val_annot.csv', data_path, xrange(7000, 8000), overwrite)

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


def create_generators(train_annotations, val_annotations, class_mapping, preprocess_image, batch_size, data_augmentation=False, base_dir=None):
    if data_augmentation:
        transform_generator = kr_train.random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = kr_train.random_transform_generator(flip_x_chance=0.5)

    # create the generators
    train_generator = CSVGenerator(
        train_annotations,
        class_mapping,
        transform_generator=transform_generator,
        base_dir=base_dir,
        preprocess_image=preprocess_image,
        batch_size=batch_size
    )

    if val_annotations:
        validation_generator = CSVGenerator(
            val_annotations,
            class_mapping,
            base_dir=base_dir,
            preprocess_image=preprocess_image,
            batch_size=batch_size
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def create_callbacks(model, batch_size, weight_file=None, tensorboard_dir=None, snapshots_path=None,
                     backbone=None, dataset_type=None):
    callbacks = []
    if tensorboard_dir:
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    # save the model
    if snapshots_path:
        # ensure directory created first; otherwise h5py will error after epoch.
        checkpoint = ModelCheckpoint(
            os.path.join(
                snapshots_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=backbone,
                                                                    dataset_type=dataset_type)
            ),
            verbose=1,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
    else:
        if not weight_file:
            weight_file = 'retinanet_unnamed.h5'
        checkpoint = ModelCheckpoint(
            weight_file,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto'
        )

    callbacks.append(checkpoint)

    callbacks.append(ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks
