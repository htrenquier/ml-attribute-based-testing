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

# def unpickle(file, encoding='bytes'):
#     with open(file, 'rb') as f:
#         di = pickle.load(f, encoding=encoding)
#     return di
#
# batches_meta = unpickle(f"../input/cifar-10-python/cifar-10-batches-py/batches.meta", encoding='utf-8')
# label_names = batches_meta['label_names']
# batch_labels = []
# batch_images = []
#
# for n in range(1, 6):
#     batch_dict = unpickle(f"../input/cifar-10-python/cifar-10-batches-py/data_batch_{n}")
#     # Add labels to the list of batch labels
#     batch_labels.append(batch_dict[b'labels'])
#
#     # Load the images, and resize them to 10000x3x32x32
#     data = batch_dict[b'data'].reshape((10000, 3, 32, 32))
#     # Modify axis to be 10000x32x32x3, since this is the correct order for keras
#     data = np.moveaxis(data, 1, -1)
#     batch_images.append(data)
#
# labels = np.concatenate(batch_labels, axis=0)
# images = np.concatenate(batch_images, axis=0)
# test_dict = unpickle(f"../input/cifar-10-python/cifar-10-batches-py/test_batch")
# test_labels = np.array(test_dict[b'labels'])
# test_images = test_dict[b'data'].reshape((10000,3,32,32))
# test_images = np.moveaxis(test_images, 1, -1)
#
# # We normalize the input according to the methods used in the paper
# X_train = preprocess_input(images)
# y_test = to_categorical(test_labels)
#
# # We one-hot-encode the labels for training
# X_test = preprocess_input(test_images)
# y_train = to_categorical(labels)
#
#
#
#
# model = kapp.vgg16.VGG16(
#     weights=None,
#     include_top=True,
#     classes=10,
#     input_shape=(32,32,3)
# )
#
# # Expand this cell for the model summary
# model.summary()
#
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='sgd',
#     metrics=['accuracy']
# )
#
# checkpoint = ModelCheckpoint(
#     'model.h5',
#     monitor='val_acc',
#     verbose=0,
#     save_best_only=True,
#     save_weights_only=False,
#     mode='auto'
# )
#
# # Train the model
# history = model.fit(
#     x=X_train,
#     y=y_train,
#     validation_split=0.1,
#     batch_size=256,
#     epochs=30,
#     callbacks=[checkpoint],
#     verbose=1
# )

#
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
#
#
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# n = 2
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     print(xs)
#     ys = randrange(n, 0, 100)
#     print(ys)
#     zs = randrange(n, zlow, zhigh)
#     print(zs)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
#     x = 25
#     y = 56
#     z = -22
#     ax.scatter(x, y, z, c=[[0.0, 1, 0]], s=300)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

# import src.model_trainer as mt
#
# train_data_orig, test_data_orig = cifar10.load_data()
# train_data, test_data = mt.format_data(train_data_orig, test_data_orig, 10)
# training_data_len = 30000
#
# d = 3
# n = 32
# distance = [[[1 for k in xrange(d)] for j in xrange(n)] for i in xrange(n)]
# data1 = [distance for k in xrange(400)]
# data2 = [distance for k in xrange(200)]
#
# print(np.array(data1).shape)
# print(np.array(data2).shape)
# print(np.array(data1+data2).shape)
#
# a = train_data_orig[0][:training_data_len]
# b = train_data_orig[0][30000:40000]
#
# c = np.concatenate((a, b))
# print(c.shape)
# print(np.array(a+b).shape)
#
# train_data_ref = [train_data_orig[0][:training_data_len]+train_data_orig[0][30000:40000],
#                   train_data_orig[1][:training_data_len]+train_data_orig[1][30000:40000]]
# print(np.array(train_data_ref).shape)


class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        s = list(s)
        p = list(p)
        return self.isMatch_rec(s, p)

    def isMatch_rec(self, s, p):
        ans = False
        conditional_ans = False
        prev_p = ''
        while s:
            print(s)
            print(p)
            current_s = s.pop(0)
            if p:
                # print('Tries to pop')
                current_p = p.pop(0)
            else:
                return False
            if p:
                if p[0] == '*':
                    conditional_ans = self.isMatch_rec(s.insert(0, current_s), p.pop(0))

            if current_p == '*':
                # print('case *')
                while p:
                    if (p[0] == prev_p):
                        p.pop(0)
                        if s:
                            s.pop(0)
                        else:
                            return ans or conditional_ans
                    else:
                        break
                while (current_s == prev_p) or (prev_p == '.'):
                    # print('while')
                    if s:
                        current_s = s.pop(0)
                    else:
                        # print('here')
                        return (not(p) and ((current_s == prev_p) or (prev_p == '.')))
                        break
                if p:
                    p.pop(0)
            elif current_p == '.':
                # print('case .')
                prev_p = current_p
                continue
            else:
                # print('case else')
                if current_p != current_s:
                    if p:
                        if p.pop(0) != '*':
                            return False
                        else:
                            s.insert(0, current_s)
                    else:
                        return False
            prev_p = current_p
        return (ans and not (s or p)) or conditional_ans

    def removeAllNonStars(self, s, p):
        k = 1
        while k <= len(p):
            if p[-k] == '*':
                k += 2
            else:
                curr_s = s.pop()

    def isMatchLeft(self, s, p):
        while s:
            curr_s = s.pop(0)
            if p:
                curr_p = p.pop(0)
            else:
                return False
            if p:
                if p[0] != '*':
                    if curr_s == curr_p or curr_p == '.':
                        continue
                    else:
                        return False
                else:
                    return self.isMatchRight(s.insert(0, curr_s), p.insert(0, curr_p))
            else:
                return (curr_s == curr_p or curr_p == '.') and (not s)

    def isMatchRight(self, s, p):
        while s:
            curr_s = s.pop()
            if p:
                curr_p = p.pop()
            else:
                return False
            if curr_p == '*':
                return self.isMatchStarredRight(s.append(curr_s), p.append(curr_p))
            else:
                if (curr_s == curr_p or curr_p == '.'):
                    continue
                else:
                    return False
        return (not p)

    def isMatchStarredRight(self, s, p):
        # called when p[-1] == '*'
        while p.pop() == '*':
            starred_char = p.pop()
            if s:
                if s[-1] != starred_char:
                    if p:
                        if p[-1] == '*':
                            continue
                        else:
                            self.isMatchStarredLeft(s, p)
                    else:
                        return False  # or (not s)
                else:
                    # s[-1] is starred_char should check p[-1]

                    if len(p) > 1:
                        if p[-1] == '*':

            if p:
                if p[-1] == starred

    def isMatch_rec(self, s, p):
        ans = False
        conditional_ans = False
        prev_p = ''
        while s:
            print(s)
            print(p)
            current_s = s.pop(0)
            if p:
                # print('Tries to pop')
                current_p = p.pop(0)
            else:
                return False
            if p:
                if p[0] == '*':
                    conditional_ans = self.isMatch_rec(s.insert(0, current_s), p.pop(0))

            if current_p == '*':
                # print('case *')

                while (current_s == prev_p) or (prev_p == '.'):
                    # print('while')
                    if s:
                        current_s = s.pop(0)
                    else:
                        # print('here')
                        return (not (p) and ((current_s == prev_p) or (prev_p == '.'))) or conditional_ans
                        break
                if p:
                    p.pop(0)
            elif current_p == '.':
                # print('case .')
                prev_p = current_p
                continue
            else:
                # print('case else')
                if current_p != current_s:
                    if p:
                        if p.pop(0) != '*':
                            return False or conditional_ans
                        else:
                            s.insert(0, current_s)
                    else:
                        return False or conditional_ans
            prev_p = current_p
        return (ans and not (s or p)) or conditional_ans

