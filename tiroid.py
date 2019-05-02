import matplotlib as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


def min_max(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def z_score(x):
    return stats.zscore(x)


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

data = pd.read_csv('data/data_tiroid_missing.csv', names=['a','b','c','d','e','label'])
new_data = data.replace("?", np.nan)
data_median = new_data.median()
data_fill = new_data.fillna(data_median)
data_fill = data_fill.astype(float)
data_fill.plot(kind='scatter', x='a' , y='b', color='blue')
plt.show()

data_mean = np.mean(new_data)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
new_data = pd.DataFrame(imp.fit_transform(new_data))

loo = LeaveOneOut()
knn = KNeighborsClassifier(n_neighbors = 3)

#### MIN MAX
new_data_min_max = min_max(new_data, -1, 1)
print("\n\nMIN MAX")
print(new_data_min_max)
print(new_data_min_max)
dataset_value = new_data_min_max[[0, 1, 2, 3, 4]]
dataset_target = new_data[[5]].astype(int)
X = dataset_value
y = dataset_target
score_min_max = cross_val_score(knn, X, y, cv=loo)
print("\n\nLOO Min Max = ", score_min_max.mean() * 100)

#### Z Score
new_data_z_score = z_score(new_data)
print("\n\nZ Score")
print(new_data_z_score)
dataset_value = new_data_z_score
dataset_target = new_data[[5]].astype(int)
# print(dataset_target)
X = dataset_value
y = dataset_target
score_z_score = cross_val_score(knn, X, y, cv=loo)
print("\n\nLOO Z Score = ", score_z_score.mean() * 100)

#### Sigmoid
new_data_sigmoid = sigmoid(new_data)
print("\n\nSigmoid")
print(new_data_sigmoid)
dataset_value = new_data_sigmoid
dataset_target = new_data[[5]].astype(int)
# print(dataset_target)
X = dataset_value
y = dataset_target
score_sigmoid = cross_val_score(knn, X, y, cv=loo)
print("\n\nLOO Sigmoid = ", score_sigmoid.mean() * 100)