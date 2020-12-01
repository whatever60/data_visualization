from re import T
from typing import Sequence

import numpy as np


class GMM:
    def __init__(
        self,
        k: int,
        comp_names: Sequence[str],
        thres: float = 1e-5,
        max_iter: int = 1000
    ) -> None:
        self.k = k
        self.thres = thres 
        self.comp_names = comp_names
        self.max_iter = max_iter

    def _one_round(self, X):
        N = self.r.sum(axis=1)  # k
        self.mu = self.r @ X / N[:, np.newaxis]  # k * d
        self.Sigma = np.array([(self.r[k] * (X - self.mu[k]).T @ (X - self.mu[k])) for k in range(self.k)]) / N[:, np.newaxis, np.newaxis]  # k * d * d
        self.pi = N / len(X)

    def fit(self, X: Sequence[Sequence]) -> None:
        np.random.shuffle(X)
        init = np.random.randint(self.k, size=len(X))
        self.r = np.eye(self.k)[:, init]
        self._one_round(X)

        for _ in range(self.max_iter):
            r = np.array([[self._calc_rnk(x, k) for x in X] for k in range(self.k)])
            self.r = r / r.sum(axis=0)  # k * n 列归一化
            mu = self.mu
            self._one_round(X)
            if np.linalg.norm(mu - self.mu) < self.thres:
                break
        else:
            print(f'Did not converge to {self.thres} after {self.max_iter} iterations. Consider tuning paras.')

    def predict(self, X):
        probas = [sum(self._calc_rnk(x, k) for k in range(self.k)) for x in X]
        return np.array(probas)

    def _calc_rnk(self, x, kth_component) -> float:
        numerator = self.pi[kth_component] * multivariate_normal(
            x,
            self.mu[kth_component],
            self.Sigma[kth_component]
        )
        return numerator



def multivariate_normal(x, mu, Sigma):
    '''多元高斯分布'''
    # print(len(X))
    if not len(x) == len(mu):
        raise ValueError(f'bad dimension, X.shape = {x.shape}, mu.shape = {mu.shape}')
    exponent = - 1 / 2 * ((x - mu) @ np.linalg.inv(Sigma) @ (x - mu).T)
    coef = (2 * np.pi) ** (- len(x) / 2) * np.linalg.det(Sigma) ** (- 1 / 2)
    return coef * np.exp(exponent)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from sklearn import mixture

    
    n_samples1 = 300
    n_samples2 = 50

    # generate random sample, two components
    np.random.seed(0)

    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples1, 2) + np.array([20, 20])

    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples2, 2), C) + np.array([-5, -5])

    # concatenate the two datasets into the final training set
    X_train = np.vstack([stretched_gaussian, shifted_gaussian])

    # # fit a Gaussian Mixture Model with two components
    use_sklearn = 0
    if use_sklearn:
        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
        clf.fit(X_train)
    else:
    
        clf = GMM(2, ['a', 'b'])
        clf.fit(X_train)

    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    
    if use_sklearn:
        Z = -clf.score_samples(XX)
    else:
        Z = -np.log10(clf.predict(XX))

    Z = Z.reshape(X.shape)

    CS = plt.contourf(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                    levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.savefig('./haha.jpg')
