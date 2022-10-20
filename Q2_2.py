import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('wdbc.data', header=None, sep=",")
data[1] = data[1].replace({'B': 0, 'M': 1})
test_ratio = 0.2

train_size = int(len(data)*(1-test_ratio))
data_train = data.loc[:train_size, :]

x_train = data.loc[0:train_size, 2:]
y_train = data.loc[0:train_size, 1]

x_test = data.loc[train_size:, 2:]
y_test = data.loc[train_size:, 1].to_numpy()
#%% prior P(y)
prior = []
prior = (x_train.groupby(y_train).apply(lambda x: len(x))/train_size).to_numpy()


#%%
def gaussian_probability(x_row,class_type, train_mean, train_var):
    a = np.exp((-1 / 2) * ((x_row - train_mean[class_type]) ** 2) / (2 * train_var[class_type]))
    b = np.sqrt(2 * np.pi * train_var[class_type])
    return a / b

#%%
train_mean = x_train.groupby(y_train).apply(np.mean).to_numpy()
train_var = x_train.groupby(y_train).apply(np.var).to_numpy()

np.seterr(divide = 'ignore')
predictions = []
for row in x_test.to_numpy():
    posteriors = {}
    for class_type in range(2):
        posterior = np.sum(np.log(gaussian_probability(row, class_type, train_mean, train_var))) + np.log(prior[class_type])
        posteriors[class_type] = posterior
    if posteriors[0] > posteriors[1]:
        predictions.append(0)
    else:
        predictions.append(1)
#%% accuracy
accuracy = np.sum(y_test == predictions) / len(y_test)

#%%
acc = []
for i in range(len(y_test)):
    acc.append((predictions[i] , y_test[i]))

benign_predicts = 0
benign_size = 0
for i in range(len(y_test)):
    if acc[i][1] == 0:
        benign_size += 1
    if acc[i][0] == acc[i][1] and acc[i][1] == 0:
        benign_predicts += 1
benign_accuracy = benign_predicts / benign_size
#%%
y_t = data.loc[train_size:, 1]
df_confusion = pd.crosstab(y_t.reset_index(drop = True), pd.Series(predictions))
