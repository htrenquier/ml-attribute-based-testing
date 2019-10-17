from __future__ import division
import tensorflow as tf
import os, sys, errno

ilsvrc2012_val_path = '/home/henri/Downloads/imagenet-val/'
ilsvrc2012_val_labels = '../ilsvrc2012/val_ground_truth.txt'
ilsvrc2012_path = '../ilsvrc2012/'
res_path = '../res/'
h5_path = '../res/h5/'
csv_path = '../res/csv/'
png_path = '../res/png/'
tb_path = '../res/logs/'
retinanet_h5_path = '../res/h5/retinanet/'
bdd100k_labels_path = "../../bdd100k/labels/"
bdd100k_data_path = "../../bdd100k/images/100k/"
bdd100k_val_path = "../../bdd100k/images/100k/val/"
bdd100k_train_path = "../../bdd100k/images/100k/train/"
bdd10k_data_path = "../../bdd100k/images/10k/"
bdd10k_val_path = "../../bdd100k/images/10k/val/"
bdd10k_train_path = "../../bdd100k/images/10k/train/"


def check_dirs(*paths):
    print(os.getcwd())
    for p in paths:
        try:
            os.mkdir(p)
        except OSError, e:
            if e.errno == errno.EEXIST:
                print ("Directory %s exists" % p)
            else:
                print ("Creation of the directory %s failed" % p)
        else:
            print ("Successfully created the directory %s " % p)


def init():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    os.chdir(os.path.dirname(sys.argv[0]))
    check_dirs(res_path,
               ilsvrc2012_path,
               h5_path,
               csv_path,
               png_path,
               bdd100k_labels_path,
               bdd100k_data_path,
               bdd100k_val_path,
               bdd100k_train_path)
    return sess
