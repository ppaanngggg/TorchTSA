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
            _use_const: bool = True
    ):
        assert _phi_num >= 0
        assert _theta_num >= 0
        assert _phi_num + _theta_num > 0

        # fitter params
        self.phi_num = _phi_num  # len of phi_arr
        self.theta_num = _theta_num  # len of theta_arr
        self.use_const = _use_const

        # model params
        self.phi_arr: np.ndarray = None
        self.theta_arr: np.ndarray = None
        self.const_arr: np.ndarray = None
        self.log_sigma_arr: np.ndarray = None

        # data buf for fitting
        self.arr: np.ndarray = None
        self.y_arr: np.ndarray = None
        self.x_arr: np.ndarray = None
        self.latent_arr: np.ndarray = None

    def func(self, _params: np.ndarray):
        # split params
        sigma = np.exp(_params[0])
        phi = _params[1:1 + self.phi_num]
        theta = _params[1 + self.phi_num: 1 + self.phi_num + self.theta_num]
        if self.use_const:
            mu = _params[-1]
        else:
            mu = self.const_arr

        if self.phi_num > 0:
            mu = phi.dot(self.x_arr) + mu
        if self.theta_num > 0:
            self.latent_arr[:self.theta_num] = 0.0
            self.latent_arr[self.theta_num:] = self.y_arr - mu
            arma_recursion(self.latent_arr, theta)
            mu += theta.dot(stack_delay_arr(
                self.latent_arr, self.theta_num
            ))

        loss = -logpdf(self.y_arr, mu, sigma ** 2).mean()

        return loss

    def fit(
            self, _arr: typing.Sequence[float],
            _max_iter: int = 50, _disp: bool = False,
    ):
        assert len(_arr) > self.phi_num
        assert len(_arr) > self.theta_num

        self.arr = np.array(_arr)

        # 1. const
        if self.use_const:
            self.const_arr = np.mean(self.arr, keepdims=True)
        else:
            self.const_arr = np.zeros(1)
        # 2. sigma
        if self.log_sigma_arr is None:
            self.log_sigma_arr = np.log(np.std(self.arr, keepdims=True))
        # 3. phi
        if self.phi_num > 0:
            self.phi_arr = np.zeros(self.phi_num)
        # 4. theta
        if self.theta_num > 0:
            self.theta_arr = np.zeros(self.theta_num)

        # init latent arr
        self.y_arr = self.arr[self.phi_num:]
        if self.phi_num > 0:
            self.x_arr = stack_delay_arr(self.arr, self.phi_num)
        if self.theta_num > 0:
            self.latent_arr = np.zeros(
                len(self.arr) - self.phi_num + self.theta_num
            )

        # concatenate params
        init_params = self.log_sigma_arr
        if self.phi_num > 0:
            init_params = np.concatenate((
                init_params, self.phi_arr
            ))
        if self.theta_num > 0:
            init_params = np.concatenate((
                init_params, self.theta_arr
            ))
        if self.use_const:
            init_params = np.concatenate((
                init_params, self.const_arr
            ))

        res = minimize(
            self.func, init_params, method='L-BFGS-B',
            options={'maxiter': _max_iter, 'disp': _disp},
        )
        params = res.x

        # update array
        self.log_sigma_arr = params[0:1]
        params = params[1:]
        if self.use_const:
            self.const_arr = params[-1:]
        if self.phi_num > 0:
            self.phi_arr = params[:self.phi_num]
        if self.theta_num > 0:
            self.theta_arr = params[self.phi_num:self.phi_num + self.theta_num]

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
        value = self.const_arr[0]
        if self.phi_num > 0:
            tmp_arr = arr[-self.phi_num:]
            tmp_arr = tmp_arr[::-1]
            value += self.phi_arr.dot(tmp_arr)
        if self.theta_num > 0:
            tmp_latent = latent[-self.theta_num:]
            tmp_latent = tmp_latent[::-1]
            value += self.theta_arr.dot(tmp_latent)

        return value

    def getPhis(self) -> np.ndarray:
        return self.phi_arr

    def getThetas(self) -> np.ndarray:
        return self.theta_arr

    def getConst(self) -> np.ndarray:
        return self.const_arr

    def getSigma(self) -> np.ndarray:
        return np.exp(self.log_sigma_arr)
