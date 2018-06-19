import numpy as np


class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be vaild"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获得数据集X的前n个元素"""
        assert self.n_components <= X.shape[1], "n_components must be greater then the feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            """计算单位向量"""
            return w / np.linalg.norm(w)

        def first_componet(X, inital_w, eta, n_iters=1e4, epsilon=1e-8):
            w = direction(inital_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                # 注意1：每次求单位向量
                w = direction(w)
                if abs(f(w, X) - f(last_w, X)) < epsilon:
                    break

                cur_iter = cur_iter + 1
            return w

        X_pca = demean(X)

        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_componet(X_pca, initial_w, eta)
            self.components_[i, :] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, X):
        """将给定的X映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X反映射会原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
