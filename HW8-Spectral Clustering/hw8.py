import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
import os

X = np.genfromtxt("odevler/dataset/hw08_data_set.csv", delimiter=",")
N = X.shape[0]
threshold = 1.25
R = 5

class_means = np.array([[+2.5, +2.5],
                        [-2.5, +2.5],
                        [-2.5, -2.5],
                        [+2.5, -2.5],
                        [0, 0]])

class_deviations = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+1.6, 0], [0, +1.6]]])

class_sizes = np.array([50, 50, 50, 50, 100])

plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], color="k")
plt.xlabel("X_1")
plt.ylabel("X_2")
plt.show()


## Euclidean Distance & Connectivity Matrix
euc_dist = np.array([[np.sqrt((X[j][0] - X[k][0])**2 + (X[j][1] - X[k][1])**2) for k in range(N)] for j in range(N)])

B = np.zeros((N, N), dtype=int)
for j in range(N):
    for k in range(N):
        if euc_dist[j, k] < threshold and j != k:
            B[j,k] = 1
        else:
            B[j,k] = 0

### Visualization of Connectivity Matrix

plt.figure(figsize=(10, 10))
for j in range(N):
    for k in range(N):
        if B[j, k] == 1 and j != k:
            plt.plot(np.array([X[j, 0], X[k, 0]]), np.array([X[j, 1], X[k, 1]]), linewidth=0.2, c="black")
plt.scatter(X[:, 0], X[:, 1], color='black')
plt.xlabel("X_1")
plt.ylabel("X_2")
plt.show()


# calculate D
D = np.zeros((N, N),dtype=int)
for j in range(N):
    b_count = 0
    for k in range(N):
        if B[j][k] == 1:
            b_count += 1
    D[j][j] = b_count

L_symmetric = np.eye(300) - np.matmul(np.sqrt(np.linalg.inv(D)), np.matmul(B, np.sqrt(np.linalg.inv(D))))

eigen_values, eigen_vectors = np.linalg.eig(L_symmetric)

Z = eigen_vectors[:,np.argsort(eigen_values)[1:R+1]]

initial_centroids = np.vstack([Z[29], Z[143], Z[204], Z[271], Z[277]])


def update_centroids(memberships, X):
    if memberships is None:
        centroids = initial_centroids
    else:
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

iteration = 1

while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, Z)  # updating the centroids at each iteration
    # k-means algorithm stops when centroids stop changing
    if np.all(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    iteration += 1

K = 5

def plot_current_state(centroids, memberships, data):
    plt.figure(figsize=(10, 10))
    centroids = update_centroids(memberships, X)
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    for c in range(K):
        plt.plot(data[memberships == c, 0], data[memberships == c, 1], ".", markersize=10,
                 color=cluster_colors[c])
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

plot_current_state(centroids, memberships, X)
