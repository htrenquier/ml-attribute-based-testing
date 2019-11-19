from __future__ import division

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as sk_metrics
from keras.datasets import cifar10

from keras_retinanet import models as kr_models
from keras_retinanet.bin import train as kr_train
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model

import initialise
import bdd100k_utils as bu
import data_tools as dt
import metrics
import metrics_color
import model_trainer as mt
import plotting
import tests_logging as t_log
import analyse
import json
import pickle

from datetime import datetime
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors

models = ('densenet121', 'resnet50', 'mobilenet', 'mobilenetv2', 'vgg16', 'vgg19', 'nasnet')
# models = ['densenet121', 'resnet50']
# models = ['mobilenet']
# models = ['mobilenet128_0.75'] # doesn't seem to work for retinanet

ilsvrc2012_val_path = '/home/henri/Downloads/imagenet-val/'
ilsvrc2012_val_labels = '../ilsvrc2012/val_ground_truth.txt'
h5_path = '../res/h5/'
csv_path = '../res/csv/'
tb_path = '../res/logs/'
retinanet_h5_path = '../res/h5/retinanet/'
bdd100k_labels_path = "../../bdd100k/labels/"
bdd100k_val_path = "../../bdd100k/images/100k/val/"
bdd100k_train_path = "../../bdd100k/images/100k/train/"

# ilsvrc2012_path = '../ilsvrc2012/'
# res_path = '../res/'
# png_path = '../res/png/'
# bdd100k_data_path = "../../bdd100k/images/100k/"
# bdd10k_data_path = "../../bdd100k/images/10k/"
# bdd10k_val_path = "../../bdd100k/images/10k/val/"
# bdd10k_train_path = "../../bdd100k/images/10k/train/"


def cifar_test():
    train_data, test_data = cifar10.load_data()
    for m in models:
        model0, model_name = mt.train2(m, train_data, test_data, 50, True, 'cifar10', h5_path)
        # model0, model_name = mt.train(m, 'cifar10', 50, data_augmentation=True)
        # y_predicted = predict(model0, test_data)
        acc, _, y_predicted = dt.predict_and_acc(model0, test_data)
        t_log.log_predictions(y_predicted, model_name, file_path=csv_path)
        # predicted_classes = np.argmax(y_predicted, axis=1)
        # true_classes = np.argmax(test_data[1], axis=1)
        # metrics.accuracy(predicted_classes, true_classes)


def imagenet_test():
    """
    https://gist.githubusercontent.com/maraoz/388eddec39d60c6d52d4/raw/791d5b370e4e31a4e9058d49005be4888ca98472/gistfile1.txt
    :return:
    """
    file_names, true_classes = t_log.read_ground_truth(ilsvrc2012_val_labels)
    for m in models:
        model, preprocess_func = mt.load_imagenet_model(m)
        y_predicted = dt.predict_dataset(file_names, ilsvrc2012_val_path, model, preprocess_func)
        t_log.log_predictions(y_predicted, model_name=m + '_imagenet', file_path=csv_path)
        predicted_classes = np.argmax(y_predicted, axis=1)
        metrics.accuracy(predicted_classes, true_classes)


def color_domain_test():
    all_data_orig = dt.get_data('cifar10', (0, 20000))
    g = 4
    n_images = 5
    # images_cube = ds.cifar10_color_domains(granularity=g, frequence=0.3)
    images_cube = dt.cifar10_maxcolor_domains(granularity=g)
    images_cube_sizes = np.zeros((g, g, g))
    total = 0
    for x in xrange(g):
        for y in xrange(g):
            for z in xrange(g):
                l = len(images_cube[x][y][z])
                images_cube_sizes[x][y][z] = l
                total += l
                id_list = images_cube[x][y][z][:n_images]
                if len(id_list) > 10000:
                    print(id_list)
                    c = 0
                    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
                                             subplot_kw={'xticks': (), 'yticks': ()})
                    for img_id in id_list:
                        ax = axes[c]
                        c += 1
                        ax.imshow(all_data_orig[0][img_id], vmin=0, vmax=1)
                        ax.set_title("id#" + str(img_id))
                    plt.show()
    print(images_cube_sizes)
    print('total', total)


def cifar_color_domains_test():
    for m in models:
        tr_data = dt.get_data('cifar10', (0, 20000))
        val_data = dt.get_data('cifar10', (20000, 30000))
        test_data = dt.get_data('cifar10', (30000, 60000))
        f_test_data = dt.format_data(test_data, 10)  # f for formatted

        model0, model_name0 = mt.train2(m, tr_data, val_data, 50, False, 'cifar10-2-5', path=h5_path)
    #
    # for m in models:
    #     model0, model_name = mt.train(m, 'cifar10', 50, data_augmentation=True)
        cube = metrics_color.color_domains_accuracy(model0)
        print('cube', cube)
        sizes_cube = dt.cube_cardinals(cube)
        print('Sizes', sizes_cube)


def show_ids():
    test_data = dt.get_data('cifar10', (50000, 60000))
    hard = [9746, 9840, 9853, 9901, 9910, 9923, 9924, 9926, 9960, 9982]
    easy = [9929, 9935, 9939, 9945, 9952, 9966, 9971, 9992, 9997, 9999]
    for k in easy:
        plotting.imshow(test_data[0][k])
    for k in hard:
        plotting.imshow(test_data[0][k])

    print('done')


def car_example():
    test_data = dt.get_data('cifar10', (50000, 60000))
    cars = [6983, 3678, 3170, 1591]

    cc0 = metrics_color.ColorDensityCube(resolution=4)
    cc0.feed(test_data[0][cars[0]])
    plotting.imshow(test_data[0][cars[0]])
    cc0.plot_cube()

    cc0 = metrics_color.ColorDensityCube(resolution=4)
    cc0.feed(test_data[0][cars[1]])
    plotting.imshow(test_data[0][cars[1]])
    cc0.plot_cube()


def show_distribution():
    images_cube = dt.cifar10_maxcolor_domains(granularity=4, data_range=(50000, 60000))
    region_sizes = dt.cube_cardinals(images_cube)
    cc = metrics_color.ColorDensityCube(resolution=4, cube=region_sizes)
    cc.normalize()
    cc.plot_cube()


def retinanet_tiny_train():
    labels_path = bdd100k_labels_path
    val_json = labels_path + 'bdd100k_labels_images_val.json'

    num_data = 7000
    batch_size = 1
    steps_per_epoch = np.ceil(num_data / batch_size)

    train_annot, val_annot = bu.annotate_tiny(val_json, labels_path, bdd100k_val_path, overwrite=True)
    cl_map_path = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    for m in models:
        print('Generating %s backbone...' % m)
        backbone = kr_models.backbone(m)
        weights = backbone.download_imagenet()
        print('Creating generators...')
        tr_gen, val_gen = bu.create_generators(train_annotations=train_annot,
                                               val_annotations=val_annot,
                                               class_mapping=cl_map_path,
                                               base_dir='',
                                               preprocess_image=backbone.preprocess_image,
                                               batch_size=batch_size)
        print('Creating models...')
        model, training_model, prediction_model = kr_train.create_models(backbone.retinanet, tr_gen.num_classes(), weights)
        print('Creating callbacks...')
        callbacks = bu.create_callbacks(model,
                                        batch_size,
                                        snapshots_path=retinanet_h5_path,
                                        tensorboard_dir=tb_path,
                                        backbone=m,
                                        dataset_type='bdd10k')

        print('Training...')
        training_model.fit_generator(
            generator=tr_gen,
            steps_per_epoch=steps_per_epoch,  # 10000,
            epochs=50,
            verbose=1,
            callbacks=callbacks,
            workers=1,  # 1
            use_multiprocessing=False,  # False,
            max_queue_size=10,
            validation_data=val_gen
        )


def retinanet_train(tiny=False):
    labels_path = bdd100k_labels_path
    val_json = labels_path + 'bdd100k_labels_images_val.json'
    train_json = labels_path + 'bdd100k_labels_images_train.json'
    val_annot = labels_path + 'val_annotations.csv'
    train_annot = labels_path + 'train_annotations.csv'

    num_data = 70000

    classes = bu.annotate(val_json, val_annot, labels_path, bdd100k_val_path)
    cl_map_path = bu.class_mapping(classes, output_csv=labels_path + 'class_mapping.csv')
    bu.annotate(train_json, train_annot, bdd100k_labels_path, bdd100k_train_path)

    # Hyper-parameters
    batch_size = 1
    steps_per_epoch = np.ceil(num_data / batch_size)

    for m in models:
        print('Generating %s backbone...' % m)
        backbone = kr_models.backbone(m)
        weights = backbone.download_imagenet()
        print('Creating generators...')
        tr_gen, val_gen = mt.create_generators(train_annotations=train_annot,
                                               val_annotations=val_annot,
                                               class_mapping=cl_map_path,
                                               base_dir='',
                                               preprocess_image=backbone.preprocess_image,
                                               batch_size=batch_size)
        print('Creating models...')
        model, training_model, prediction_model = kr_train.create_models(backbone.retinanet, tr_gen.num_classes(), weights)
        print('Creating callbacks...')
        callbacks = mt.create_callbacks(model, batch_size, 'test', tensorboard_dir=tb_path)

        print('Training...')
        training_model.fit_generator(
            generator=tr_gen,
            steps_per_epoch=steps_per_epoch,  # 10000,
            epochs=2,
            verbose=1,
            callbacks=callbacks,
            workers=4,  # 1
            use_multiprocessing=True,  # False,
            max_queue_size=10,
            validation_data=val_gen
        )


def retinanet_test():
    test_images = ['fe1b92a1-faaaa1eb.jpg',
                   'fe1cc363-a3f36598.jpg',
                   'fe1d74f0-5cdc4057.jpg',
                   'fe1d74f0-6969bdb5.jpg',
                   'fe1d9184-cd999efe.jpg',
                   'fe1d9184-d144106a.jpg',
                   'fe1d9184-dec09b65.jpg',
                   'fe1f2409-5b415eb7.jpg',
                   'fe1f2409-c16ea1ed.jpg',
                   'fe1f55fa-19ba3600.jpg']

    model = kr_models.load_model(retinanet_h5_path+'densenet121_bdd10k_18.h5', backbone_name='densenet121')
    kr_models.check_training_model(model)
    inference_model = kr_models.convert_model(model)
    labels_to_names = bu.get_label_names(bdd100k_labels_path + 'class_mapping.csv')
    # load image
    image = read_image_bgr(bdd100k_val_path+test_images[2])

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = inference_model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


def box_size():
    labels_path = bdd100k_labels_path
    val_annot = labels_path + 'val_annotations.csv'
    train_annot = labels_path + 'train_annotations.csv'

    print('val:')
    x_avg, y_avg, n = bu.avg_box_size(val_annot)
    print('x_avg', x_avg, 'y_avg', y_avg, 'n', n)
    print('train:')
    x_avg, y_avg, n = bu.avg_box_size(train_annot)
    print('x_avg', x_avg, 'y_avg', y_avg, 'n', n)


def test_do_boxes_cross():
    # x_min1, y_min1, x_max1, y_max1
    boxes = [
              (391, 378, 401, 390),
              (371, 408, 398, 431),
              (113, 334, 136, 350),
              (426, 376, 449, 427),
              (606, 310, 631, 353),
              (491, 388, 511, 444),
              (583, 384, 611, 431),
              (405, 408, 415, 429),
              (366, 386, 378, 401)
              ]

    for i in xrange(len(boxes)-1):
        for j in xrange(i+1, len(boxes)):
            print(bu.do_boxes_collide(boxes[i], boxes[j]))


def test_extract_non_superposing_boxes():
    labels_path = bdd100k_labels_path
    val_annot = labels_path + 'val_annotations.csv'
    train_annot = labels_path + 'train_annotations.csv'
    val_obj_annot = labels_path + 'obj_val_annotations.csv'
    train_obj_annot = labels_path + 'obj_train_annotations.csv'
    bu.annotate_objects(val_annot, val_obj_annot)
    bu.annotate_objects(train_annot, train_obj_annot)


def check_obj_annotations(obj_annot_file=bdd100k_labels_path + 'obj_val_annotations.csv'):
    # labels_path = bdd100k_labels_path
    # train_obj_annot = labels_path + 'obj_train_annotations.csv'
    with open(obj_annot_file) as obj_annot:
        for k in xrange(5):
            line = obj_annot.readline()
            curr = str(line.split(',')[0])
            boxes = []
            labels = []
            format = (64, 64)
            img_size = (1280, 720)
            while line and curr == str(line.split(',')[0]):
                box = bu.adjust_position(
                        bu.adjust_size(
                            bu.adjust_ratio(
                                bu.get_box(line), format), format), img_size)
                boxes.append(box)
                labels.append(line.split(',')[-1].rstrip())
                line = obj_annot.readline()
            # load image
            image = read_image_bgr(curr)
            # copy to draw on
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            for box, label in zip(boxes, labels):
                # scores are sorted so we can break
                color = label_color(label)
                draw_box(draw, box, color=color)
                if label == 'car':
                    caption = "{} {:.0f}x{:.0f}".format(label, int(box[2])-int(box[0]), int(box[3])-int(box[1]))
                else:
                    caption = 'o'
                draw_caption(draw, box, caption)

            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(draw)
            plt.show()


def classification_dataset():
    bdd100k_cl_val_path = '../../bdd100k/classification/images/val/'
    bdd100k_cl_train_path = '../../bdd100k/classification/images/train/'
    bdd100k_cl_val_path = '../../bdd100k/classification/images/val/'

    # bu.build_dataset('../../bdd100k/labels/train_annotations.csv',
    #                  bdd100k_cl_train_path,
    #                  '../../bdd100k/classification/labels/train_ground_truth.csv',
    #                      make_attributes_file=True)

    # labels_path = '../../bdd100k/labels/'
    #
    # bu.annotate_attributes(bdd100k_labels_path + 'bdd100k_labels_images_val.json',
    #                        bdd100k_labels_path + 'bdd100k_labels_images_val_attributes.csv',
    #                        bdd100k_cl_val_path, overwrite=False)

    # bu.annotate_attributes(bdd100k_labels_path + 'bdd100k_labels_images_train.json',
    #                        bdd100k_labels_path + 'bdd100k_labels_images_train_attributes.csv',
    #                        bdd100k_cl_train_path, overwrite=False)

    # bu.build_dataset('../../bdd100k/labels/val_annotations.csv',
    #                  bdd100k_cl_val_path,
    #                  '../../bdd100k/classification/labels/val_ground_truth.csv',
    #                  make_attributes_file=True)


def train_bdd100k_cl():
    labels_path = '../../bdd100k/classification/labels/'
    train_labels = '../../bdd100k/classification/labels/train_ground_truth.csv'
    val_labels = '../../bdd100k/classification/labels/val_ground_truth.csv'
    # class_map_file = labels_path + 'class_mapping.csv'
    val_json = '../../bdd100k/labels/bdd100k_labels_images_val.json'

    epochs = 20

    # Parameters
    params = {'dim': (64, 64, 3),
              'batch_size': 32,
              'n_classes': 10,
              'shuffle': True}

    class_map_file = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    # Datasets
    val_partition, val_labels = bu.get_ids_labels(val_labels, class_map_file)
    tr_partition, tr_labels = bu.get_ids_labels(train_labels, class_map_file)

    # Generators
    training_generator = mt.DataGenerator(tr_partition[:500000], tr_labels, **params)
    validation_generator = mt.DataGenerator(val_partition[:100000], val_labels, **params)
    print(len(training_generator))

    for m in models:

        weight_file = mt.weight_file_name(m, 'bdd100k_cl0-500k', epochs, data_augmentation=False)
        weight_file = h5_path + weight_file
        print("Building: " + weight_file)
        if m in ('mobilenet', 'mobilenetv2', 'nasnet'):
            ###
            model = mt.model_struct(m, (224, 224, 3), params['n_classes'], weights='imagenet', include_top=False)
            new_model = mt.model_struct(m, params['dim'], params['n_classes'], weights=None, include_top=False)
            print("Loading weights...")

            for new_layer, layer in zip(new_model.layers[1:], model.layers[1:]):
                new_layer.set_weights(layer.get_weights())
            base_model = new_model
            ###
        else:
            base_model = mt.model_struct(m, params['dim'], params['n_classes'], weights='imagenet', include_top=False)

        print("Configuring top layers")
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        # for layer in base_model.layers:
        #     layer.trainable = False

        model.compile('adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(weight_file.rstrip('.h5')+'_ep{epoch:02d}_vl{val_loss:.2f}.hdf5',
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto')

        # Train model on dataset
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            verbose=1,
                            epochs=epochs,
                            use_multiprocessing=True,
                            workers=6,
                            callbacks=[checkpoint]
                            )


def load_model_test():
    labels_path = '../../bdd100k/classification/labels/'
    # train_labels = '../../bdd100k/classification/labels/train_ground_truth.csv'
    val_labels_csv = '../../bdd100k/classification/labels/val_ground_truth.csv'
    # class_map_file = labels_path + 'class_mapping.csv'
    val_json = '../../bdd100k/labels/bdd100k_labels_images_val.json'

    # Parameters
    params = {'dim': (64, 64, 3),
              'batch_size': 32,
              'n_classes': 10,
              'shuffle': False}

    n_test_data = 100000

    class_map_file = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    # Datasets
    val_partition, val_labels = bu.get_ids_labels(val_labels_csv, class_map_file)

    # Generators
    validation_generator = mt.DataGenerator(val_partition[:n_test_data], val_labels, **params)

    label_distrib = [val_labels.values()[:n_test_data].count(k)/n_test_data for k in xrange(params['n_classes'])]
    print(label_distrib)

    # model_files = ['densenet121_bdd100k_cl0-500k_20ep_woda_ep16_vl0.95.hdf5']
    model_files = ['densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22.hdf5',
                   'resnet50_bdd100k_cl0-500k_20ep_woda_ep13_vl0.27.hdf5',
                   'mobilenet_bdd100k_cl0-500k_20ep_woda_ep15_vl0.24.hdf5',
                   'mobilenetv2_bdd100k_cl0-500k_20ep_woda_ep17_vl0.22.hdf5',
                   'nasnet_bdd100k_cl0-500k_20ep_woda_ep17_vl0.24.hdf5']

    for model_file in model_files:
        start_time = datetime.now()
        m = load_model(h5_path + model_file)
        print('File successfully loaded', model_file, 'in', str(datetime.now() - start_time))

        print("Validation ")
        start_time = datetime.now()
        # print(m.metrics_names)
        # print(m.evaluate_generator(validation_generator))
        # print('Model successfully evaluated', model_file, 'in (s)', str(datetime.now() - start_time))

        print('Writing predictions')
        predictions_file = '.'.join(model_file.split('.')[:-1])+'_predictions.csv'
        out_pr = open(csv_path + predictions_file, 'w')

        start_time = datetime.now()
        y_predicted = m.predict_generator(validation_generator)

        # prediction
        for i in xrange(len(y_predicted)):
            out_pr.write(val_partition[i] + ',' + str(y_predicted[i].tolist())+'\n')
        out_pr.close()

        predicted_classes = np.argmax(y_predicted, axis=1)

        print('Predictions successfully written', model_file, 'in', str(datetime.now() - start_time))
        true_classes = [val_labels[id] for id in val_partition[:len(y_predicted)]]
        acc = metrics.accuracy(predicted_classes, true_classes)
        print('acc=', acc)
        print(sk_metrics.confusion_matrix(true_classes, predicted_classes))
    # m.summary()


def bdd100k_sel_partition_test():
    labels_path = '../../bdd100k/classification/labels/'
    train_labels = '../../bdd100k/classification/labels/train_ground_truth.csv'
    val_json = '../../bdd100k/labels/bdd100k_labels_images_val.json'

    class_map_file = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    # Datasets
    tr_partition, tr_labels = bu.get_ids_labels(train_labels, class_map_file)
    ft_partition = tr_partition[500000:1000000]
    sel_partition = analyse.bdd100k_analysis('densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22.hdf5', ft_partition, 300000)

    print('selection res=', len(sel_partition))


def bdd100k_global_finetune_test():
    labels_path = '../../bdd100k/classification/labels/'
    train_labels = '../../bdd100k/classification/labels/train_ground_truth.csv'
    val_labels_csv = '../../bdd100k/classification/labels/val_ground_truth.csv'
    # class_map_file = labels_path + 'class_mapping.csv'
    val_json = '../../bdd100k/labels/bdd100k_labels_images_val.json'

    # Parameters
    params = {'dim': (64, 64, 3),
              'batch_size': 32,
              'n_classes': 10,
              'shuffle': False}

    n_test_data = 100000
    epochs = 30

    class_map_file = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    # Datasets
    tr_partition, tr_labels = bu.get_ids_labels(train_labels, class_map_file)
    val_partition, val_labels = bu.get_ids_labels(val_labels_csv, class_map_file)

    # label_distrib = [val_labels.values()[:n_test_data].count(k)/n_test_data for k in xrange(params['n_classes'])]
    # print(label_distrib)

    model_files = ['densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22.hdf5',
                   # ]
                   'resnet50_bdd100k_cl0-500k_20ep_woda_ep13_vl0.27.hdf5',
                   'mobilenet_bdd100k_cl0-500k_20ep_woda_ep15_vl0.24.hdf5',
                   'mobilenetv2_bdd100k_cl0-500k_20ep_woda_ep17_vl0.22.hdf5',
                   'nasnet_bdd100k_cl0-500k_20ep_woda_ep17_vl0.24.hdf5']

    for model_file in model_files:
        ft_partition = tr_partition[500000:1000000]
        n_sel_data = 300000
        sel_partition = analyse.bdd100k_analysis(model_file, ft_partition, n_sel_data)  # Selected data partition

        # Generators
        finetune_generator = mt.DataGenerator(sel_partition[:n_sel_data], tr_labels, **params)
        reference_generator = mt.DataGenerator(tr_partition[500000:500000+len(ft_partition)], tr_labels, **params)
        validation_generator = mt.DataGenerator(val_partition[:n_test_data], val_labels, **params)

        # finetune
        model = load_model(h5_path + model_file)
        model.compile('adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(model_file.rstrip('.hdf5') + '_ftep{epoch:02d}_vl{val_loss:.2f}.hdf5',
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto')

        # Train model on selected dataset
        ft_history = model.fit_generator(generator=finetune_generator,
                                         validation_data=validation_generator,
                                         verbose=1,
                                         epochs=epochs,
                                         use_multiprocessing=True,
                                         workers=6,
                                         callbacks=[checkpoint]
                                         )

        with open(model_file.rstrip('.hdf5') + '_ft_hist.pkl', 'w') as fd:
            pickle.dump(ft_history, fd)

        # reference
        model = load_model(h5_path + model_file)
        model.compile('adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(model_file.rstrip('.h5') + '_refep{epoch:02d}_vl{val_loss:.2f}.hdf5',
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto')

        # Train model on ref dataset
        ref_history = model.fit_generator(generator=reference_generator,
                                          validation_data=validation_generator,
                                          verbose=1,
                                          epochs=epochs,
                                          use_multiprocessing=True,
                                          workers=6,
                                          callbacks=[checkpoint]
                                          )

        with open(model_file.rstrip('.hdf5') + '_ref_hist.pkl', 'w') as fd:
            pickle.dump(ref_history, fd)


def bdd100k_local_finetune_test():
    labels_path = '../../bdd100k/classification/labels/'
    train_labels = '../../bdd100k/classification/labels/train_ground_truth.csv'
    val_labels_csv = '../../bdd100k/classification/labels/val_ground_truth.csv'
    # class_map_file = labels_path + 'class_mapping.csv'
    val_json = '../../bdd100k/labels/bdd100k_labels_images_val.json'

    # Parameters
    params = {'dim': (64, 64, 3),
              'batch_size': 32,
              'n_classes': 10,
              'shuffle': False}

    # n_test_data = 100000
    epochs = 30

    class_map_file = bu.class_mapping(input_json=val_json, output_csv=labels_path + 'class_mapping.csv')

    # Datasets
    tr_partition, tr_labels = bu.get_ids_labels(train_labels, class_map_file)
    val_partition, val_labels = bu.get_ids_labels(val_labels_csv, class_map_file)

    # label_distrib = [val_labels.values()[:n_test_data].count(k)/n_test_data for k in xrange(params['n_classes'])]
    # print(label_distrib)

    model_files = ['densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22.hdf5',
                   ]
                   # 'resnet50_bdd100k_cl0-500k_20ep_woda_ep13_vl0.27.hdf5',
                   # 'mobilenet_bdd100k_cl0-500k_20ep_woda_ep15_vl0.24.hdf5',
                   # 'mobilenetv2_bdd100k_cl0-500k_20ep_woda_ep17_vl0.22.hdf5',
                   # 'nasnet_bdd100k_cl0-500k_20ep_woda_ep17_vl0.24.hdf5']

    for model_file in model_files:
        ft_partition = tr_partition[500000:1000000]
        n_sel_data = 400000
        # Selected data partition
        day_sel_partition = analyse.bdd100k_analysis(model_file, ft_partition, n_sel_data, 'timeofday', 'day')
        night_sel_partition = analyse.bdd100k_analysis(model_file, ft_partition, n_sel_data, 'timeofday', 'night')

        # Generators
        day_ft_generator = mt.DataGenerator(day_sel_partition[:300000], tr_labels, **params)
        night_ft_generator = mt.DataGenerator(night_sel_partition[:300000], tr_labels, **params)
        day_val_generator = mt.DataGenerator(day_sel_partition[300000:], val_labels, **params)
        night_val_generator = mt.DataGenerator(night_sel_partition[300000:], val_labels, **params)

        # finetune
        model = load_model(h5_path + model_file)
        model.compile('adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(model_file.rstrip('.hdf5') + '_ft_day_ep{epoch:02d}_vl{val_loss:.2f}.hdf5',
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto')

        # Train model on selected dataset
        day_history = model.fit_generator(generator=day_ft_generator,
                                         validation_data=day_val_generator,
                                         verbose=1,
                                         epochs=epochs,
                                         use_multiprocessing=True,
                                         workers=6,
                                         callbacks=[checkpoint]
                                         )

        with open(model_file.rstrip('.hdf5') + '_ft_day_hist.pkl', 'w') as fd:
            pickle.dump(day_history, fd)

        # reference
        model = load_model(h5_path + model_file)
        model.compile('adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(model_file.rstrip('.h5') + '_ft_night_ep{epoch:02d}_vl{val_loss:.2f}.hdf5',
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto')

        # Train model on ref dataset
        night_history = model.fit_generator(generator=night_ft_generator,
                                          validation_data=night_val_generator,
                                          verbose=1,
                                          epochs=epochs,
                                          use_multiprocessing=True,
                                          workers=6,
                                          callbacks=[checkpoint]
                                          )

        with open(model_file.rstrip('.hdf5') + '_ft_night_hist.pkl', 'w') as fd:
            pickle.dump(night_history, fd)


def show_history_test(filename):
    with open(filename, 'r') as pkl_fd:
        history = pickle.load(pkl_fd)
    plotting.plot_history(history, 'acc', 'Densenet121 accuracy during training')


def adjectives_finding_test():
    words = ['traffic', 'sign', 'light', 'car', 'rider', 'motor', 'person', 'bus', 'truck', 'bike', 'train']
    print("loading w2vec")
    modelName = '../../GoogleNews-vectors-negative300.bin'
    w2v_model = KeyedVectors.load_word2vec_format(modelName, binary=True)
    print("w2vec loaded")
    similar_words = w2v_model.most_similar(words, topn=20)
    print(similar_words)
    # words = ['amazing', 'interesting', 'love', 'great', 'nice']
    # pos_all = dict()
    # for w in similar_words:
    #     pos_l = set()
    #     for tmp in wn.synsets(w):
    #         if tmp.name().split('.')[0] == w:
    #             pos_l.add(tmp.pos())
    #     pos_all[w] = pos_l
    # print pos_all
    for w in words:
        tmp = wn.synsets(w)[0].pos()
        print w, ":", tmp


def subset_selection_test():
    labels_path = '../../bdd100k/classification/labels/'
    bdd100k_labels_path = "../../bdd100k/labels/"
    val_labels_csv = '../../bdd100k/classification/labels/val_ground_truth.csv'
    class_map_file = labels_path + 'class_mapping.csv'
    val_json = '../../bdd100k/labels/bdd100k_labels_images_val.json'
    attr_val_file = bdd100k_labels_path + 'bdd100k_labels_images_val_attributes.csv'
    attr_tr_file = bdd100k_labels_path + 'bdd100k_labels_images_train_attributes.csv'
    train_labels = '../../bdd100k/classification/labels/train_ground_truth.csv'

    class_map_file = bu.class_mapping(input_json=val_json, output_csv=class_map_file)

    # Dataset for analysis
    tr_partition, tr_labels = bu.get_ids_labels(train_labels, class_map_file)

    w, s, tod, wst_dk2ak = bu.wst_attribute_mapping(attr_tr_file)

    d_tod = analyse.DiscreteAttribute(tod)
    d_s = analyse.DiscreteAttribute(s)

    scene_tod_distrib = np.zeros((len(d_tod.get_labels()), len(d_s.get_labels())))
    print(scene_tod_distrib)
    for data_key in tr_partition:
        attr_key = wst_dk2ak(data_key)
        x = d_tod.index_of(d_tod.labelof(attr_key))
        y = d_s.index_of(d_s.labelof(attr_key))
        scene_tod_distrib[x][y] += 1

    print("        " + " / ".join(d_s.get_labels()))
    for k in xrange(len(scene_tod_distrib)):
        print(k)
        print(d_tod.get_labels()[k] + "\t" + " ".join([str(val) for val in scene_tod_distrib[k]]))


def main():
    initialise.init()
    # Tests
    # test()
    # show_distribution()
    # retinanet_train()
    # retinanet_tiny_train()
    # retinanet_test()

    # imagenet_test()
    # cifar_color_domains_test()
    # show_ids()
    # box_size()
    # test_do_boxes_cross()
    # check_obj_annotations()
    # test_extract_non_superposing_boxes()
    # classification_dataset()

    # bdd100k_sel_partition_test()

    # bdd100k_global_finetune_test()
    bdd100k_local_finetune_test()
    # show_history_test(tb_path+'densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22_ft_hist.pkl')
    # show_history_test(tb_path + 'densenet121_bdd100k_cl0-500k_20ep_woda_ep20_vl0.22_ref_hist.pkl')
    # subset_selection_test()

    # adjectives_finding_test()
    # train_bdd100k_cl()
    # load_model_test()



main()
