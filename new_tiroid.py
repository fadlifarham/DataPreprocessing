import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

from scipy import stats

def knn(train_data, train_label, test_data, test_label):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_data, train_label.values.ravel())

    predicted = model.predict(test_data)
    # print(predicted[0])
    # print(test_label['label'].values[0])

    if predicted[0] != test_label['label'].values[0]:
        return True
    return False

def sigmoid_logic(data):
    return 1. / (1. + np.exp(-data))

def sigmoid(data, label):
    n = data.shape[0]
    error = 0
    for i in range(0, n):
        train_data = data.drop(data.index[i])
        train_label = label.drop(data.index[i])
        train_data = sigmoid_logic(train_data)

        test_data = data.iloc[[i]]
        test_label = label.iloc[[i]]
        test_data = sigmoid_logic(test_data)

        do_knn = knn(train_data, train_label, test_data, test_label)
        if (do_knn):
            error += 1
    print(train_data)
    print(test_data)
    sum_error = error / n
    print("Error Sigmoidal : ", str(sum_error), " %")

def z_score(data, label):
    n = data.shape[0]
    error = 0
    for i in range(0, n):
        train_data = data.drop(data.index[i])
        train_label = label.drop(data.index[i])
        train_data = stats.zscore(train_data, axis=1, ddof=1)

        test_data = data.iloc[[i]]
        test_label = label.iloc[[i]]
        test_data = stats.zscore(test_data, axis=1, ddof=1)

        do_knn = knn(train_data, train_label, test_data, test_label)
        if (do_knn):
            error += 1
    print(train_data)
    print(test_data)
    sum_error = error / n
    print("Error Z Score : ", str(sum_error), " %")
    

def minmax(data, label):
    scaler = MinMaxScaler()

    n = data.shape[0]

    error = 0
    for i in range(0, n):
        train_data = data.drop(data.index[i])
        train_label = label.drop(data.index[i])

        scaler.fit(train_data)
        train_data = scaler.transform(train_data)

        test_data = data.iloc[[i]]
        test_label = label.iloc[[i]]
        test_data = scaler.transform(test_data)

        do_knn = knn(train_data, train_label, test_data, test_label)
        if (do_knn):
            error += 1
    print(train_data)
    print(test_data)
    sum_error = error / n
    print("Error MinMax : " + str(sum_error) + " %")

def imputasi(data):
    data_label_list = data['label'].unique().tolist()

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    fill_data = pd.DataFrame()
    for i in range(0, len(data_label_list)):
        data_label = data.loc[data['label'] == data_label_list[i]]
        mean_data = pd.DataFrame(imp.fit_transform(data_label))

        fill_data = fill_data.append(mean_data)

    fill_data.columns = ['a','b','c','d','e','label']
    return fill_data

def main():
    fields = ['a','b','c','d','e','label']
    data = pd.read_csv('data/data_tiroid_missing.csv', names=fields)

    ### --- Original Data
    print("\n\nData Before Imputation")
    print(data)
    input("Do You Want To Continue? ")

    ### --- Imputasi Missing Value ---
    nan_data = data.replace("?", np.nan)
    data_imputation = imputasi(nan_data)
    
    the_label = data_imputation.loc[:, 'label':]
    the_label = the_label.astype(int)

    the_data = data_imputation.loc[:, :'e']
    the_data = the_data.astype(float)
    print("\n\nData After Imputation")
    print(the_data)
    input("Do You Want To Continue? ")

    ### --- Do MinMax
    print("\n\nDo Minimax :")
    minmax(the_data, the_label)
    input("Do You Want To Continue? ")

    ### --- Do Z Score
    print("\n\nDo Z Score :")
    z_score(the_data, the_label)
    input("Do You Want To Continue? ")

    ### --- Do Sigmoidal
    print("\n\nDo Sigmoidal :")
    sigmoid(the_data, the_label)

if __name__ == "__main__":
    main()