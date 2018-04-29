import sys
sys.path.append('../')

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
num_training = 10000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 1000
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

from cs231n.classifiers import KNearestNeighbor

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
X_train_folds = np.split(X_train, num_folds)
y_train_folds = np.split(y_train, num_folds)
k_to_accuracies = {}

for k_choice in k_choices:
    for i in range(num_folds):
        knn = KNearestNeighbor()
        xtrain = X_train_folds[:i] + X_train_folds[i+1:]
        xtrain = np.asarray([item for sublist in xtrain for item in sublist])
        ytrain = y_train_folds[:i] + y_train_folds[i+1:]
        ytrain = np.asarray([item for sublist in ytrain for item in sublist])
        knn.train(xtrain, ytrain)
        dists = knn.compute_distances_no_loops(np.asarray(X_train_folds[i]))
        y_test_pred = knn.predict_labels(dists, k=k_choice)
        num_correct = np.sum(y_test_pred == y_train_folds[i])
        accuracy = float(num_correct) / len(y_train_folds[i])
        k_to_accuracies.setdefault(k_choice,[]).append(accuracy)
        print('k = %d, accuracy = %f' % (k_choice, accuracy))

for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


print(k_choices[np.argmax(accuracies_mean)])
