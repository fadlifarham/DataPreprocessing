import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def minmax(data, label):
    # le = preprocessing.LabelEncoder()
    scaler = MinMaxScaler()
    # scaler.fit(data)
    # new_data = scaler.transform(data)
    # return new_data

    
    # print(data.shape[0])
    n = data.shape[0]
    error = 0
    # temp = data.drop(0)
    # print(temp)
    # print(data.value(0))
    for i in range(0, n):
        train_data = data.drop(i)
        train_label = label.drop(i)

        scaler.fit(train_data)
        train_data = scaler.transform(train_data)

        test_data = data.iloc[[i]]
        test_label = label.iloc[[i]]
        # print(test_data['a'])
        test_data = scaler.transform(test_data)

        # # train_data_encoded = le.
        # test_data_encoded = le.fit_transform(test_data)
        

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(train_data, train_label.values.ravel())

        predicted = model.predict(test_data)
        # print(predicted[0])
        # print(test_label['label'].values[0])

        if predicted[0] != test_label['label'].values[0]:
            error += 1
        
        # print(train_data)
        # print(test_data)
        # print()

        # print(train_label)
        # print(test_label)

        # break
    print(error)
    sum_error = error / n
    print(sum_error)

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
    # fill_data = fill_data.astype(float)
    # print("\n\n")
    # print("Imputasi Missing Value")
    # print(the_label)

    ### --- Do MinMax
    minmax(the_data, the_label)

if __name__ == "__main__":
    main()