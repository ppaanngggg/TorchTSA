import typing

import numpy as np
from TorchTSA.utils.recursions import arma_recursion
from scipy.optimize import minimize

from TorchTSA.utils.math import logpdf
from TorchTSA.utils.op import stack_delay_arr


class ARMAModel:

    def __init__(
            self,
            _phi_num: int = 1,
            _theta_num: int = 1,
            _use_mu: bool = True
    ):
        assert _phi_num >= 0
        assert _theta_num >= 0
        assert _phi_num + _theta_num > 0

        # fitter params
        self.phi_num = _phi_num  # len of phi_arr
        self.theta_num = _theta_num  # len of theta_arr
        self.use_mu = _use_mu
        self.split_index = np.cumsum((
            1, self.phi_num, self.theta_num,
        ))

        # model params
        self.phi_arr: np.ndarray = None
        self.theta_arr: np.ndarray = None
        self.mu_arr: np.ndarray = None
        self.log_sigma_arr: np.ndarray = None

        # data buf for fitting
        self.arr: np.ndarray = None
        self.y_arr: np.ndarray = None
        self.x_arr: np.ndarray = None
        self.latent_arr: np.ndarray = None

    def func(self, _params: np.ndarray):
        # split params
        if self.use_mu:
            mu = _params[-1]
        else:
            mu = self.mu_arr
        sigma, phi, theta, _ = np.split(_params, self.split_index)
        sigma = np.exp(sigma)

        tmp = mu
        if self.phi_num > 0:
            tmp += phi.dot(self.x_arr)
        if self.theta_num > 0:
            self.latent_arr[:self.theta_num] = 0.0
            self.latent_arr[self.theta_num:] = self.y_arr - tmp
            arma_recursion(self.latent_arr, theta)
        else:
            self.latent_arr = self.y_arr - tmp

        loss = -logpdf(
            self.latent_arr[self.theta_num:], 0, sigma ** 2
        ).mean()

        return loss

    def fit(
            self, _arr: typing.Sequence[float],
            _max_iter: int = 50, _disp: bool = False,
    ):
        assert len(_arr) > self.phi_num
        assert len(_arr) > self.theta_num

        # init latent arr
        self.arr = np.array(_arr)
        self.y_arr = self.arr[self.phi_num:]
        if self.phi_num > 0:
            self.x_arr = stack_delay_arr(self.arr, self.phi_num)
        self.latent_arr = np.empty(
            len(self.arr) - self.phi_num + self.theta_num
        )

        # 1. const
        if self.use_mu:
            self.mu_arr = np.mean(self.arr, keepdims=True)
        else:
            self.mu_arr = np.zeros(1)
        # 2. sigma
        if self.log_sigma_arr is None:
            self.log_sigma_arr = np.log(np.std(self.arr, keepdims=True))
        # 3. phi
        if self.phi_arr is None:
            self.phi_arr = np.zeros(self.phi_num)
        # 4. theta
        if self.theta_arr is None:
            self.theta_arr = np.zeros(self.theta_num)

        # concatenate params
        init_params = np.concatenate((
            self.log_sigma_arr, self.phi_arr, self.theta_arr,
        ))
        if self.use_mu:
            init_params = np.concatenate((init_params, self.mu_arr))

        res = minimize(
            self.func, init_params, method='L-BFGS-B',
            options={'maxiter': _max_iter, 'disp': _disp},
        )
        params = res.x

        # update array
        self.log_sigma_arr, self.phi_arr, self.theta_arr, _ = np.split(
            params, self.split_index
        )
        if self.use_mu:
            self.mu_arr = params[-1:]

    def predict(
            self,
            _arr: typing.Sequence[float],
            _latent: typing.Sequence[float] = None,
    ) -> float:
        arr = np.array(_arr)
        if _latent is None:
            latent = self.latent_arr
        else:
            latent = np.array(_latent)
        value = self.mu_arr[0]
        if self.phi_num > 0:
            tmp_arr = arr[-self.phi_num:]
            tmp_arr = tmp_arr[::-1]
            value += self.phi_arr.dot(tmp_arr)
        if self.theta_num > 0:
            tmp_latent = latent[-self.theta_num:]
            tmp_latent = tmp_latent[::-1]
            value += self.theta_arr.dot(tmp_latent)

        return value

    def getLatent(self) -> typing.Union[None, np.ndarray]:
        if self.theta_num > 0:
            return self.latent_arr[self.theta_num:]
        else:
            return None

    def getPhis(self) -> np.ndarray:
        return self.phi_arr

    def getThetas(self) -> np.ndarray:
        return self.theta_arr

    def getMu(self) -> np.ndarray:
        return self.mu_arr

    def getSigma(self) -> np.ndarray:
        return np.exp(self.log_sigma_arr)
