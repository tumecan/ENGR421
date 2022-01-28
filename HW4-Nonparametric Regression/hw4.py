import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
print(os.getcwd())
os.chdir(r'\Users\tumec\PycharmProjects\kocuni')



## İmport data
data = pd.read_csv("odevler/dataset/hw04_data_set.csv")

data.shape

## divide data train and test sest
train_data = data[0:150]
test_data = data[150:]

## Split data train and test set
x_train = train_data.iloc[:,0]
y_train = train_data.iloc[:,1]
x_test = test_data.iloc[:,0]
y_test = test_data.iloc[:,1]



## Define parameters
bin_width = 0.37
origin = 1.5
minimum_value = 1.5
maximum_value = max(x_train)
data_interval = np.linspace(minimum_value, maximum_value, 1601)
N = data.shape[0]

## Estimate border
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)


## Plotting
plt.figure(figsize = (10, 6))

plt.scatter(x_train, y_train, c='b',s =30, label='training')
plt.scatter(x_test, y_test, c='r',s = 30, label='test')
plt.xlabel("Eruptiontime (mm)")
plt.ylabel("Waiting timeto next eruption (min)")
plt.legend(loc='upper left')


## Estimate regressor
g = [np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])) * y_train) / np.sum(
        (left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))]


for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g[b], g[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g[b], g[b + 1]], "k-")
plt.show()


### Calculation RMSE
rmse= np.sqrt(sum([((y_test[(left_borders[i] < x_test) & (x_test <= right_borders[i])] - g[i])**2).sum() for i in range(len(left_borders))]) /len(x_test))
print("Regressogram => RMSE is " ,rmse , "when h is" , bin_width)



### Running mean smoother
mean_smooth= [np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width)))*y_train) / np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) for x in data_interval]


left_borders = data_interval[:-1]
right_borders = data_interval[1:]

### Calculation RMSE for mean smoother
rmse= np.sqrt(sum([((y_test[(left_borders[i] < x_test) & (x_test <= right_borders[i])] - mean_smooth[i])**2).sum() for i in range(len(left_borders))]) /len(x_test))

## Plot mean smoother
plt.figure(figsize = (10, 6))

plt.scatter(x_train, y_train, c='b',s =30, label='training')
plt.scatter(x_test, y_test, c='r',s = 30, label='test')
plt.xlabel("Eruption time (mm)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left')

plt.plot(data_interval, mean_smooth, "k-")

plt.show()

print("Mean Smoother => RMSE is " ,rmse , "when h is" , bin_width)

### Calculating kernel

mean = [(np.sum(
    1 / np.sqrt(2 * np.pi) * np.exp((x - x_train) * (-x + x_train) / (2 * bin_width ** 2 )) * (
        y_train)) / np.sum(
    1 / np.sqrt(2 * np.pi) * np.exp((x - x_train) * (-x + x_train) / (2 * bin_width ** 2))))  for x in data_interval]



## Drawing Kernel Smoother
plt.figure(figsize = (10, 6))
plt.scatter(x_train, y_train, c='b',s =30, label='training')
plt.scatter(x_test, y_test, c='r',s = 30, label='test')
plt.xlabel("Eruption time (mm)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left')

plt.plot(data_interval, mean, "k-")

plt.show()

### Calculation RMSE for kernel
rmse= np.sqrt(sum([((y_test[(left_borders[i] < x_test) & (x_test <= right_borders[i])] - mean[i])**2).sum() for i in range(len(left_borders))]) /len(x_test))

print("Kernel => RMSE is " ,rmse , "when h is" , bin_width)