import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def minmax(data, label):
    scaler = MinMaxScaler()
    n = data.shape[0]
    error = 0

    for i in range(0, n):
        train_data = data.drop(i)
        train_label = label.drop(i)

        scaler.fit(train_data)
        train_data = scaler.transform(train_data)

        test_data = data.iloc[[i]]
        test_label = label.iloc[[i]]
        test_data = scaler.transform(test_data)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(train_data, train_label.values.ravel())

        predicted = model.predict(test_data)

        if predicted[0] != test_label['label'].values[0]:
            error += 1
    # print(error)
    sum_error = error / n
    print("Error MinMax : " + str(sum_error) + " %")

def main():
    fields = ['a','b','c','d','e','label']
    data = pd.read_csv('data/data_tiroid_missing.csv', names=fields)

    ### --- Imputasi Missing Value ---
    nan_data = data.replace("?", np.nan)
    median = nan_data.median()
    fill_data = nan_data.fillna(median)
    the_data = fill_data.loc[:, :'e']
    the_data = the_data.astype(float)
    the_label = fill_data.loc[:, 'label':]
    the_label = the_label.astype(int)

    ### --- Do MinMax
    minmax(the_data, the_label)

if __name__ == "__main__":
    main()