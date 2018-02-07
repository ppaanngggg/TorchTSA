import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from TorchTSA.utils.op import stack_delay_arr_T


class VARModel:
    def __init__(
            self, _length: int, _phi_num: int,
            _use_mu: bool = True,
    ):
        # fitter params
        self.length = _length
        self.phi_num = _phi_num
        self.use_mu = _use_mu
        self.split_index = np.cumsum((
            self.length ** 2, self.phi_num * self.length ** 2
        ))

        # model params
        self.phi_arr: np.ndarray = None
        self.cholesky_cov_arr: np.ndarray = None
        self.mu_arr: np.ndarray = None

        # data buf for fitting
        self.arr: np.ndarray = None
        self.y_arr: np.ndarray = None
        self.x_arr: np.ndarray = None

    def func(self, _params: np.ndarray):
        if self.use_mu:
            mu = _params[-self.length:]
        else:
            mu = self.mu_arr
        cov, phi, _ = np.split(_params, self.split_index)
        cov = cov.reshape(self.length, self.length)
        phi = phi.reshape(self.phi_num, self.length, self.length)

        tmp = self.y_arr - np.matmul(phi, self.x_arr).sum(0)
        loss = -multivariate_normal.logpdf(
            tmp.T, mu, cov.dot(cov.T)
        ).mean()
        return loss

    def fit(
            self, _arr: np.ndarray,
            _max_iter: int = 100, _disp: bool = False,
    ):
        self.arr = _arr
        self.y_arr = self.arr[self.phi_num:].T
        self.x_arr = stack_delay_arr_T(self.arr, self.phi_num)

        # 1. mu
        if self.use_mu:
            self.mu_arr = np.mean(self.arr, axis=0)
        else:
            self.mu_arr = np.zeros(self.length)
        # 2. cov
        if self.cholesky_cov_arr is None:
            self.cholesky_cov_arr = np.linalg.cholesky(np.cov(self.arr.T))
        # 3. phi
        if self.phi_arr is None:
            self.phi_arr = np.zeros((self.phi_num, self.length, self.length))

        init_params = np.concatenate((
            self.cholesky_cov_arr.flatten(), self.phi_arr.flatten()
        ))
        if self.use_mu:
            init_params = np.concatenate((init_params, self.mu_arr))

        res = minimize(
            self.func, init_params, method='L-BFGS-B',
            options={'maxiter': _max_iter, 'disp': _disp},
        )
        params = res.x

        self.cholesky_cov_arr, self.phi_arr, _ = np.split(
            params, self.split_index
        )
        self.cholesky_cov_arr = self.cholesky_cov_arr.reshape(
            self.length, self.length
        )
        self.phi_arr = self.phi_arr.reshape(
            self.phi_num, self.length, self.length
        )
        if self.use_mu:
            self.mu_arr = params[-self.length:]

    def getPhis(self) -> np.ndarray:
        return self.phi_arr

    def getMu(self) -> np.ndarray:
        return self.mu_arr

    def getCov(self) -> np.ndarray:
        return self.cholesky_cov_arr.dot(self.cholesky_cov_arr.T)
