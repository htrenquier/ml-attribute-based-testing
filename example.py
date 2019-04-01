import tensorflow as tf
from keras.datasets import cifar10
import keras.applications as kapp
from keras import utils
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#
#
# # Example train VGG16 with Keras-Applications on cifar10, no data augmentation, 50 epochs
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# input_shape = x_train.shape[1:]
# model = kapp.vgg16.VGG16(include_top=True,
#                          weights=None,
#                          input_tensor=None,
#                          input_shape=input_shape,
#                          pooling=None,
#                          classes=10)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = utils.to_categorical(y_train, 10)
# y_test = utils.to_categorical(y_test, 10)
#
# model.fit(x_train,
#           y_train,
#           batch_size=32,
#           epochs=50,
#           validation_data=(x_test, y_test),
#           verbose=2,
#           shuffle=True)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint
import pickle

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# input_shape = x_train.shape[1:]
# print(input_shape)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = utils.to_categorical(y_train, 10)
# y_test = utils.to_categorical(y_test, 10)

def unpickle(file, encoding='bytes'):
    with open(file, 'rb') as f:
        di = pickle.load(f, encoding=encoding)
    return di

batches_meta = unpickle(f"../input/cifar-10-python/cifar-10-batches-py/batches.meta", encoding='utf-8')
label_names = batches_meta['label_names']
batch_labels = []
batch_images = []

for n in range(1, 6):
    batch_dict = unpickle(f"../input/cifar-10-python/cifar-10-batches-py/data_batch_{n}")
    # Add labels to the list of batch labels
    batch_labels.append(batch_dict[b'labels'])

    # Load the images, and resize them to 10000x3x32x32
    data = batch_dict[b'data'].reshape((10000, 3, 32, 32))
    # Modify axis to be 10000x32x32x3, since this is the correct order for keras
    data = np.moveaxis(data, 1, -1)
    batch_images.append(data)

labels = np.concatenate(batch_labels, axis=0)
images = np.concatenate(batch_images, axis=0)
test_dict = unpickle(f"../input/cifar-10-python/cifar-10-batches-py/test_batch")
test_labels = np.array(test_dict[b'labels'])
test_images = test_dict[b'data'].reshape((10000,3,32,32))
test_images = np.moveaxis(test_images, 1, -1)

# We normalize the input according to the methods used in the paper
X_train = preprocess_input(images)
y_test = to_categorical(test_labels)

# We one-hot-encode the labels for training
X_test = preprocess_input(test_images)
y_train = to_categorical(labels)




model = kapp.vgg16.VGG16(
    weights=None,
    include_top=True,
    classes=10,
    input_shape=(32,32,3)
)

# Expand this cell for the model summary
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_acc',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

# Train the model
history = model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.1,
    batch_size=256,
    epochs=30,
    callbacks=[checkpoint],
    verbose=1
)