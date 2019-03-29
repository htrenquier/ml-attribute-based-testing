import tensorflow as tf
from keras.datasets import cifar10
import keras.applications as kapp

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
model = kapp.vgg16.VGG16(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=input_shape,
                         pooling=None,
                         classes=10)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=50,
          validation_data=(x_test, y_test),
          verbose=2,
          shuffle=True)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

