
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import os
import scipy.stats as stats
import pandas as pd
from math import e, pow
print(os.getcwd())
os.chdir(r'\Users\tumec\PycharmProjects\kocuni')




class_means = np.array([[+2.5, +2.5],
                        [-2.5, +2.5],
                        [-2.5, -2.5],
                        [+2.5, -2.5],
                        [0, 0]])  # class means already given
class_deviations = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+1.6, 0], [0, +1.6]]])

class_sizes = np.array([50, 50, 50, 50, 100])


## Importing Data Set
X = np.genfromtxt("odevler/dataset/hw07_data_set.csv", delimiter=',')
centroids_1 = np.genfromtxt("odevler/dataset/hw07_initial_centroids.csv", delimiter=',')

#X.shape

N = X.shape[0]
K = 5

# data points plotted

plt.plot(X[:, 0], X[:, 1],".",color="k" )
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


def update_centroids(memberships, X):
    if memberships is None:
        # centroids initialized from given csv file
        centroids = np.genfromtxt("odevler/dataset/hw07_initial_centroids.csv", delimiter=',')
    else:
        centroids = np.vstack([np.mean(X[memberships == k,], axis=0) for k in range(K)])  # updated centroids
    return centroids

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


## Visualization

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12,
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")

centroids = None
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, X)
    if np.alltrue(centroids == old_centroids):
        break
    else:
        plt.figure(figsize = (12, 6))
        plt.subplot(1, 2, 1)
        plot_current_state(centroids, memberships, X)

    old_memberships = memberships
    memberships = update_memberships(centroids, X)
    if np.alltrue(memberships == old_memberships):
        plt.show()
        break
    else:
        plt.subplot(1, 2, 2)
        plot_current_state(centroids, memberships, X)
        plt.show()

    iteration = iteration + 1

print(centroids)


# one-hot encoding for the memberships set
membership = np.zeros((N, K))
for n in range(N):
    for k in range(K):
        if memberships[n] == k + 1:
            membership[n][k] = 1

memberships = membership

D = 2
sample_covariances = [np.eye(D) for k in range (K)]
priors = [class_sizes[k] / K for k in range(K)]
sample_means = centroids
covariances = sample_covariances


X_matrix = np.asmatrix(X)
h_ik = np.zeros((N, K))


# EM algorithm
for i in range(0, 100):

    for k in range(N):
        sum = 0
        for j in range(K):
            xx = np.matrix([X[k] - sample_means[j]])
            mat = xx.dot(np.linalg.inv(covariances[j])).dot(xx.T)
            mat = mat * (-.5)
            h_ik[k][j] = priors[j] * pow(np.linalg.det(covariances[j]), -0.5) * pow(e, mat[0])
            sum += h_ik[k][j]
        h_ik[k] /= sum

    sample_means = h_ik.T.dot(X_matrix)
    tempHsum = np.sum(h_ik, axis=0)

    sample_means = sample_means / tempHsum[:, None]
    sample_means = np.asarray(sample_means)
    covariances = []

    for j in range(K):
        sum = 0
        for k in range(N):
            xx = np.matrix([X[k] - sample_means[j]])
            mat = xx.T.dot(xx)
            result = mat * h_ik[k][j]
            sum += result
        sum /= tempHsum[j]
        covariances.append(sum)

# Mean vectors of EM algorithm
print(sample_means)
