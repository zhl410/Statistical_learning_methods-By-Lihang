import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.001, max_iter=None):
        """
        learning_rate: 学习率
        max_iter: 最大迭代次数
        """
        self.max_iter = 10000
        self.tol = 1e-3
        self.learning_rate = learning_rate
        self.b = 0.0
        self.w = None
        
    def sign(self, arr):
        return np.sign((arr>0)-0.5)
    def fit(self, X_train, y_train):
        self.w = np.zeros(len(X_train[0]))
        loss = []
		"""
		书中实现的是随机梯度下降，每次只对迭代一个样本
		这里全批量梯度下降，一个batch就是所有的样本。
		向量化操作，更加简洁。
		"""
        for epoch in range(self.max_iter):
            function_margin = y_train*(np.dot(X_train, self.w) + self.b)
            error_points = (function_margin <= 0)
            loss.append(-np.sum(function_margin[error_points]))
            self.w += self.learning_rate * np.dot(y_train[error_points], X_train[error_points])
            self.b += self.learning_rate * np.sum(y_train[error_points]) 
            #print("epoch%d:, loss%f"%(epoch, loss[epoch]))
            
        print("trianing done")
    def score(self, X_val, y_val):
        y_pre =  self.predit(X_val)
        acc = np.sum(y_pre == y_val) / len(y_val)
        print(acc)
        return acc
    def predit(self, X_test):
        y_pre = self.sign(np.dot(X_test, self.w) + self.b)
        return y_pre