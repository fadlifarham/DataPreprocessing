import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_file(name_file):
    fields = ['x', 'y', 'value']
    reader = pd.read_csv(name_file, names=fields)
    return reader

def main():
    file = read_file("data/ruspini.csv")
    x = file['x'].values
    y = file['y'].values
    plt.scatter(x, y)
    # plt.show()

    file2 = read_file("data/data_ruspini_missing.csv")
    x2 = file2['x'].values
    y2 = file2['y'].values
    plt.scatter(x2, y2)
    # plt.show()

    # file2['x'][0] = 0
    # print(file['x'].sum())
    file2['x'] = file2['x'].replace(['?'], 0)
    file2['y'] = file2['y'].replace(['?'], 0)

    # print(file2.shape[0])

    dx = file2['x'].sum() #/ file2.shape[0]
    dy = file2['y'].sum() #/ file2.shape[0]

    print(str(dx))
    print(str(dy))


    # print(file2)

if __name__ == "__main__":
    main()