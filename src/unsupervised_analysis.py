import data_tools as dt
import tests_logging as t_log
import metrics
from keras.datasets import cifar10
from sklearn.cluster import DBSCAN
from sklearn import metrics as sk_metrics
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt


file_list = ['densenet121_cifar10-2-5_50ep_woda_ft50ep-exp-res.csv',
             'densenet121_cifar10-2-5_50ep_woda_ft50ep-ref-res.csv']
res_path = '../res/'

train_data_orig, test_data_orig = cifar10.load_data()
test_imgs = test_data_orig[0]
test_labels = [int(test_data_orig[1][k][0]) for k in xrange(len(test_data_orig[1]))]
len_data = len(test_labels)
X=[]
image = test_imgs[0]

for img in test_imgs:
    # image = (img[0]+img[1]+img[2])/(3*255)
    image = img/255
    X.append(image.flatten())
X = np.array(X)
image_shape = X[0].shape
print(image_shape)

pca = PCA(n_components=200, whiten=True, random_state=0)
pca.fit_transform(X)
X_pca = pca.transform(X)

n_clusters = 10


def agglo_test():
    # Agglomerative
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agglomerative.fit_predict(X_pca)
    print("Cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))
    return labels_agg


def k_means_test():
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_pca)
    labels_kmeans = kmeans.labels_
    print("Cluster sizes kmeans: {}".format(np.bincount(labels_kmeans)))
    return labels_kmeans


def dbscan_test():
    # DBSCAN
    db = DBSCAN(eps=10, min_samples=10)
    labels_dbscan = db.fit_predict(X)
    print('Unique labels: {}' .format(np.unique(labels_dbscan)))
    print('Cluster sizes: {}' .format(np.bincount(labels_dbscan + 1)))
    return labels_dbscan


def lfw_people_test():
    """
    From /Introduciton to Machine Learning with Python A.C. Muller & S. Guido/.
    :return:
    """
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    print('image_shape' + str(image_shape))
    print('people.images.shape: {}'.format(people.images.shape))
    print('Number of classes: {}'.format(len(people.target_names)))

    counts = np.bincount(people.target)

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255
    print(X_people.shape)

    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    db = DBSCAN(eps=7, min_samples=3)
    labels = db.fit_predict(X_people)
    print('Unique labels: {}' .format(np.unique(labels)))
    print('Cluster sizes: {}' .format(np.bincount(labels + 1)))

    for cluster in range(max(labels) + 1):
        mask = labels == cluster
        n_images = int(np.sum(mask))
        fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
                             subplot_kw={'xticks':(), 'yticks':()})
        for image, label, ax in zip(X_people[mask], y_people[mask], axes):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1])
        plt.show()

labels = [[] for k in xrange(n_clusters)]
for id, x in enumerate(k_means_test()):
    labels[x].append(id)


for f in file_list[:1]:
    predictions = aa.load_csv(res_path + f, 2)
    losses = t_log.load_csv(res_path + f, 3)
    pr = metrics.prediction_ratings(losses, test_labels)
    sorted_pr_indexes = np.argsort(pr)

    per_class_score = np.zeros(10)
    guessed = []

    for id, p in enumerate(predictions):
        if p == test_labels[id]:
            per_class_score[p] += 1
            guessed.append(True)
        else:
            guessed.append(False)

    print(per_class_score)
    print(np.mean(pr))

    for l in xrange(len(labels)):
        n_images = min(10, len(labels[l]))
        print(n_images)
        print(labels[l])
        if n_images > 1:
            fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
                                     subplot_kw={'xticks': (), 'yticks': ()})
            for i in xrange(min(10, n_images)):
                ax = axes[i]
                img_id = labels[l][i]
                ax.imshow(test_imgs[img_id], vmin=0, vmax=1)
                fig.suptitle('label #' + str(l) + ' (' + str(n_images) + '/' + str(len(labels[l])) + ' images)')
            plt.show()


    n_images = 10
    n_rows = 10
    # for th in xrange(n_rows):
    #     fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
    #                              subplot_kw={'xticks': (), 'yticks': ()})
    #     for dec in xrange(n_images):
    #         ax = axes[dec]
    #         pr_rank = th*1000+dec
    #         img_id = sorted_pr_indexes[pr_rank]
    #         print(str(pr_rank) + ': ' + str(pr[img_id]) + ' conf. guessed = ' + str(guessed[img_id]))
    #         ax.imshow(test_imgs[img_id], vmin=0, vmax=1)
    #         ax.set_title('pr #' + str(pr_rank) + '\nds #' + str(img_id))
    #     plt.show()
    #     print(th)


    # #DBSCAN
    # noise_ids = []
    # noise_ranks = []
    # for id,l in enumerate(labels):
    #     if l == -1:
    #         noise_ids.append(id)
    #         noise_ranks.append(np.where(sorted_pr_indexes == id)[0][0])
    #
    # print(np.mean(noise_ranks))
    # print(noise_ranks)

    # Agglomerative clustering

    print('#############################################################')
    for cl in labels:
        noise_ids = []
        noise_ranks = []
        for l, id in enumerate(cl):
            noise_ids.append(id)
            index = np.where(sorted_pr_indexes == id)[0][0]
            noise_ranks.append(index)
        print(np.mean(noise_ranks))
        # print('cluster = ' + str(cl))
        # print(noise_ranks)

    # cluster = labels[3]
    # print(cluster)
    # n_images = len(cluster)
    # fig, axes = plt.subplots(1, n_images, figsize=(n_images, 4),
    #                          subplot_kw={'xticks': (), 'yticks': ()})
    # for k in xrange(n_images):
    #     ax = axes[k]
    #     print(ax)
    #     id = cluster[k]
    #     print(id)
    #     ax.imshow(test_imgs[id], vmin=0, vmax=1)

        # ax.set_title(people.target_names[label].split()[-1])
    plt.show()
    # gg_indexes = []  # good guesses
    # gg_imgs = pd.DataFrame([])
    # bg_indexes = []  # bad guesses
    # bg_imgs = pd.DataFrame([])
    # predictions = aa.load_csv(res_path+f, 2)
    # for i in xrange(len_data):
    #     if predictions[i] == test_labels[i]:
    #         gg_indexes.append(i)
    #         img = test_imgs[i]/255
    #         img = pd.Series(img.flatten(), name=str(i))
    #         gg_imgs = gg_imgs.append(img)
    #     else:
    #         bg_indexes.append(i)
    #         img = test_imgs[i] / 256
    #         img = pd.Series(img.flatten(), name=str(i))
    #         bg_imgs = bg_imgs.append(img)
    #
    # X = gg_imgs
    # print(len(gg_indexes))
    # print(len(gg_imgs))
    # print(len(bg_indexes))
    # print(len(bg_imgs))
    #
    # print(np.array(X).shape)

