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
