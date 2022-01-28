import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.linalg as linalg
import matplotlib.pyplot as plt

clothes = pd.read_csv(r"odevler\dataset\hw02_images.csv", header=None)
labels = pd.read_csv(r"odevler\dataset\hw02_labels.csv", header=None)

# clothes = np.genfromtxt(r"odevler\dataset\hw02_images.csv", delimiter=",")
# labels = np.genfromtxt(r"odevler\dataset\hw02_labels.csv", delimiter=",", dtype=int)

# clothes.shape
# labels.shape

clothes_data = pd.concat([clothes, labels], axis=1)
# clothes_data = np.hstack((clothes, labels[:, None]))
# clothes_data.shape

X = clothes_data.iloc[:, :-1]
# X.shape
# X.head()

y = clothes_data.iloc[:, -1]
# y.shape
# y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14285, random_state=42)
# X_train.shape

K = np.max(y_train)
N = X_train.shape[0]
D = X_train.shape[1]

sample_means = np.array([X_train[y_train[:] == 1].mean().tolist(),
                         X_train[y_train[:] == 2].mean().tolist(),
                         X_train[y_train[:] == 3].mean().tolist(),
                         X_train[y_train[:] == 4].mean().tolist(),
                         X_train[y_train[:] == 5].mean().tolist()])

print(sample_means)


sample_deviations = [np.sqrt(((X_train[y_train[:] == 1] - sample_means[0]) ** 2).mean().tolist()),
                     np.sqrt(((X_train[y_train[:] == 2] - sample_means[1]) ** 2).mean().tolist()),
                     np.sqrt(((X_train[y_train[:] == 3] - sample_means[2]) ** 2).mean().tolist()),
                     np.sqrt(((X_train[y_train[:] == 4] - sample_means[3]) ** 2).mean().tolist()),
                     np.sqrt(((X_train[y_train[:] == 5] - sample_means[4]) ** 2).mean().tolist())]
print(sample_deviations)

class_priors = [np.mean(y_train[:] == i) for i in range(1, K + 1)]
print(class_priors)

def predict(train):
    scores = []
    for i in range(K):
        scores.append(sum((-0.5 * np.log(2 * math.pi)) - (np.log(sample_deviations[i])) - (
                (train - sample_means[i]) ** 2 / (2 * sample_deviations[i] ** 2))) + np.log(class_priors[i]))
    score_series = pd.Series(scores)
    return score_series[score_series == np.max(score_series)].index[0] + 1

def get_pred_labels(data):
    return np.array([predict(data.iloc[i, :]) for i in range(data.shape[0])])


y_pred_train = get_pred_labels(X_train)
confusion_matrix_train = pd.crosstab(y_pred_train, y_train, rownames=["y_pred_train"], colnames=["y_truth"])

y_pred_test = get_pred_labels(X_test)
confusion_matrix_test = pd.crosstab(y_pred_test, y_test, rownames=["y_pred_test"], colnames=["y_truth"])
