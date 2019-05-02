import matplotlib.pyplot as plt

with open("data/data_ruspini_missing.csv") as f:
    data = []
    for line in f:
        temp = []
        for num in line.split(","):
            if num == '?':
                temp.append(0)
            else:
                temp.append(int(num))
        data.append(temp)

x = []
y = []
for item in data:
    x.append(item[0])
    y.append(item[1])


plt.scatter(x, y)

plt.show()