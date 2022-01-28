import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(17)

# mean parameters
means = np.array([[0.0, 2.5],
                        [-2.5, -2.0],
                        [2.5, -2.0]])

# covariance parameters
covariances = np.array([[[3.2, 0],
                               [0, 1.2]],
                              [[1.2, 0.8],
                               [0.8, 1.2]],
                              [[1.2, -0.8],
                              [-0.8, 1.2]]])
# sample sizes
sizes = np.array([120, 80,100])

# generate random samples
points1 = np.random.multivariate_normal(means[0,:], covariances[0,:,:], sizes[0])
points2 = np.random.multivariate_normal(means[1,:], covariances[1,:,:], sizes[1])
points3 = np.random.multivariate_normal(means[2,:], covariances[2,:,:], sizes[2])
X = np.vstack((points1, points2,points3 ))

# generate corresponding labels
y = np.concatenate((np.repeat(1, sizes[0]), np.repeat(2, sizes[1]), np.repeat(3, sizes[2])))

# plot data points generated
plt.figure(figsize = (10, 10))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


data = np.hstack((X, y[:, None]))

X = data[:,[0, 1]]
y_truth = data[:,2].astype(int)

K = np.max(y_truth)
N = data.shape[0]

# calculate sample means
sample_means = [np.mean(X[y == (c + 1)], axis = 0) for c in range(K)]
sample_means

# calculate sample deviations
sample_covariances= [(np.matmul(np.transpose(X[y == (c + 1)]-sample_means[c]),(X[y == (c + 1)] - sample_means[c])))/sizes[c] for c in range(K)]
sample_covariances

# calculate prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
class_priors


# Calculate Parameters
def w_function(sample_covariances, sample_means, class_priors):
    W = [-0.5 * np.linalg.inv(sample_covariances[c]) for c in range(K)]
    Wc = [np.transpose(np.matmul(np.linalg.inv(sample_covariances[c]), sample_means[c])) for c in range(K)]
    Wc_0 = [-0.5 * np.matmul(np.matmul((sample_means[c]), np.linalg.inv(sample_covariances[c])),
                             np.transpose(sample_means[c]))
            - 0.5 * np.log(np.linalg.det(sample_covariances[c])) + np.log(class_priors[c]) for c in range(K)]

    return W, Wc, Wc_0

def score(X,sample_covariances,sample_means,class_priors):
    W, Wc, Wc_0 = w_function(sample_covariances,sample_means,class_priors)
    scores = np.array([0, 0, 0])
    for i in range(K):
        score1 = np.matmul(np.matmul(np.transpose(X), W[i]), X)
        score2 = score1 + np.matmul(np.transpose(Wc[i]), X)
        score3 = score2 + Wc_0[i]
        scores[i] = score3
    return scores

y_predicted = np.argmax([score(X[i],sample_covariances,sample_means,class_priors) for i in range(len(X))], axis = 1)+1
y_predicted

## Confusion Matrix
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames=['y_predicted'], colnames=['y_truth'])
confusion_matrix


########## 5
"""
x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))



plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values, levels = 0, colors = "k")




plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
"""