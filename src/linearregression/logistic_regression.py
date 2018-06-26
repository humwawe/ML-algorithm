import numpy as np

from src.common.metrics import accuracy_score


class LogisticRegression:

    def __init__(self):
        """初始化Logistic Regression模型"""

        # 系数向量（θ1,θ2,.....θn）
        self.coef_ = None
        # 截距 (θ0)
        self.interception_ = None
        # θ向量
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train,y_train, 使用梯度下降法训练Logistic Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

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

    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.coef_ is not None and self.interception_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
