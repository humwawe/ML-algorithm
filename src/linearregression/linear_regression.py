import numpy as np

from src.common.metrics import r2_score


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""

        # 系数向量（θ1,θ2,.....θn）
        self.coef_ = None
        # 截距 (θ0)
        self.interception_ = None
        # θ向量
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train，y_train 训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        # np.ones((len(X_train), 1)) 构造一个和X_train 同样行数的，只有一列的全是1的矩阵
        # np.hstack 拼接矩阵
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # X_b.T 获取矩阵的转置
        # np.linalg.inv() 获取矩阵的逆
        # dot() 矩阵点乘
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train,y_train, 使用梯度下降法训练Linear Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            #
            # for i in range(1, len(theta)):
            #     res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:, i]))
            res = X_b.T.dot(X_b.dot(theta) - y)

            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=n_iters, epsilon=1e-8):
            """
            梯度下降法封装
            X_b: X特征矩阵
            y: 结果向量
            initial_theta:初始化的theta值
            eta:学习率η
            n_iters: 最大循环次数
            epsilon: 精度
            """
            theta = initial_theta
            i_iters = 0

            while i_iters < n_iters:
                """
                如果theta两次变化之间的损失函数值的变化小于我们定义的精度,则可以说明我们已经找到了最低的损失函数值和对应的theta

                如果循环次数超过了我们设置的循环次数，则说明可能由于η设置的过大导致无止境的循环
                """
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                i_iters += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """
        根据训练数据集X_train, y_train, 使用随机梯度下降法训练Linear Regression模型
        :param X_train:
        :param y_train:
        :param n_iters: 在随机梯度下降法中，n_iters代表所有的样本会被看几圈
        :param t0:
        :param t1:
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            """
            去X_b,y 中的随机一个元素进行导数公式的计算
            :param theta:
            :param X_b_i:
            :param y_i:
            :return:
            """
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):

            def learning_rate(t):
                """
                计算学习率，t1 为了减慢变化速度，t0为了增加随机性
                :param t: 第t次循环
                :return:
                """
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                # 对X_b进行一个乱序的排序
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]

                # 对整个数据集看一遍
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.coef_ is not None and self.interception_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
