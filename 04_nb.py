#Compute the log of the sum of exponentials of input elements
from scipy.special import  logsumexp
#View inputs as arrays with at least two dimensions.
#np.atleast_2d
from sklearn import datasets
#from sklearn.naive_bayes import GaussianNB
import numpy as np
from abc import ABCMeta, abstractmethod

class BaseNB(metaclass=ABCMeta):
    
    @abstractmethod
    def _joint_log_likelihood(self, X):
        """
        计算没有被归一化的后验概率的对数 I.e. ``log P(c) + log P(x|c)``, 右边分子部分
        Returns: [n_sample, n_classes]
        """
        pass
    def predict(self, X):
        """
        X : array-like, shape = [n_samples, n_features]
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        X : array-like, shape = [n_samples, n_features]
        Returns : array-like, shape = [n_samples, n_classes]
        """
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        X : array-like, shape = [n_samples, n_features]
        Returns : array-like, shape = [n_samples, n_classes]
        """
        return np.exp(self.predict_log_proba(X))
    
class GaussianNB(BaseNB):
    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
    def _get_mean_variance(self, X, sample_weight=None):
        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X, axis=0, weights=sample_weight / n_new)
            new_var = np.average((X - new_mu) ** 2, axis=0,
                                 weights=sample_weight / n_new)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        return new_mu, new_var
    def fit(self, X, y, sample_weight=None):
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)

        for (i, y_i) in enumerate(self.classes_):
            X_i = X[y == y_i, :]
            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]
            self.theta_[i, :], self.sigma_[i, :] = self._get_mean_variance(X_i, sw_i)
            self.class_count_[i] += N_i
        self.sigma_[:, :] += self.epsilon_

        # Set priors 
        if self.priors:
            self.class_prior_ = np.asarray(self.priors)
        else:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self
    
    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood
    
iris = datasets.load_iris()
gnb = GaussianNB(priors=[.3, .2, .5])
gnb.fit(iris.data, iris.target)
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#gnb.predict_proba(iris.data)
(y_pred == iris.target).sum() / len(iris.target)