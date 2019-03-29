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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)


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
    x=x_train,
    y=y_train,
    validation_split=0.1,
    batch_size=256,
    epochs=30,
    callbacks=[checkpoint],
    verbose=1
)