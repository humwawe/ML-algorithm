from collections import Counter
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    raw_data_x = [[3.393533211, 2.331273381],
                  [2.110073483, 1.781539638],
                  [1.343808831, 3.368360954],
                  [3.582294042, 4.679179110],
                  [2.280362439, 2.866990263],
                  [7.423436942, 4.696522875],
                  [5.745051997, 3.533989803],
                  [9.172168622, 2.511101045],
                  [7.792783481, 3.424088941],
                  [7.939820817, 0.791637231]
                  ]
    raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    X_train = np.array(raw_data_x)
    y_train = np.array(raw_data_y)
    # 要预测的点
    x = np.array([8.093607318, 3.365731514])
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='r')
    plt.scatter(x[0], x[1], color='b')
    # plt.show()
    distances = []
    # for x_train in X_train:
    #     d = sqrt(np.sum((x_train - x) ** 2))
    #     distances.append(d)
    distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]

    # 返回排序后的结果的索引,也就是距离测试点距离最近的点的排序坐标数组
    nearest = np.argsort(distances)

    k = 6

    topK_y = [y_train[i] for i in nearest[:k]]

    # collections的Counter方法可以求出一个数组的相同元素的个数，返回一个dict【key=元素名，value=元素个数】
    # most_common方法求出最多的元素对应的那个键值对
    votes = Counter(topK_y)
    predict_y = votes.most_common(1)[0][0]
    print(predict_y)
