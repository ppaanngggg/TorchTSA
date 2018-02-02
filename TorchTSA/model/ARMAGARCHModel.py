import typing

import numpy as np
from TorchTSA.utils.recursions import arma_recursion, garch_recursion
from scipy.optimize import minimize

from TorchTSA.utils.math import logit, ilogit, logpdf
from TorchTSA.utils.op import stack_delay_arr


class ARMAGARCHModel:
    def __init__(
            self,
            _phi_num: int = 1,
            _theta_num: int = 1,
            _alpha_num: int = 1,
            _beta_num: int = 1,
            _use_mu: bool = True,
    ):
        # fitter params
        self.phi_num = _phi_num
        self.theta_num = _theta_num
        self.alpha_num = _alpha_num
        self.beta_num = _beta_num
        self.use_mu = _use_mu
        self.split_index = np.cumsum((
            1, self.phi_num, self.theta_num,
            self.alpha_num, self.beta_num,
        ))

        # model params
        self.mu_arr: np.ndarray = None
        self.phi_arr: np.ndarray = None
        self.theta_arr: np.ndarray = None
        self.logit_alpha_arr: np.ndarray = None
        self.logit_beta_arr: np.ndarray = None
        self.log_const_arr: np.ndarray = None

        # buffers
        self.arr: np.ndarray = None
        self.y_arr: np.ndarray = None
        self.x_arr: np.ndarray = None
        self.latent_arma_arr: np.ndarray = None  # new info
        self.latent_garch_arr: np.ndarray = None  # var of new info

    def func(self, _params: np.ndarray):
        # split params
        if self.use_mu:
            mu = _params[-1]
        else:
            mu = self.mu_arr
        const, phi, theta, alpha, beta, _ = np.split(
            _params, self.split_index
        )
        const = np.exp(const)
        alpha = ilogit(alpha)
        beta = ilogit(beta)

        # arma part
        if self.phi_num > 0:
            mu = phi.dot(self.x_arr) + mu
        if self.theta_num > 0:
            self.latent_arma_arr[:self.theta_num] = 0.0
            self.latent_arma_arr[self.theta_num:] = self.y_arr - mu
            arma_recursion(self.latent_arma_arr, theta)
        else:
            self.latent_arma_arr = self.y_arr - mu

        # garch part
        new_info_arr = self.latent_arma_arr[self.theta_num:]
        square_arr = new_info_arr ** 2
        arch_x_arr = stack_delay_arr(square_arr, self.alpha_num)
        if self.beta_num > 0:  # estimate latent_arr
            self.latent_garch_arr[self.beta_num:] = alpha.dot(
                arch_x_arr) + const
            self.latent_garch_arr[:self.beta_num] = square_arr.mean()
            garch_recursion(self.latent_garch_arr, beta)
        else:
            self.latent_garch_arr = alpha.dot(arch_x_arr) + const

        loss = -logpdf(
            new_info_arr[self.alpha_num:], 0, self.latent_garch_arr[self.beta_num:]
        ).mean()

        return loss

    def optimize(
            self, _init_params: np.ndarray,
            _max_iter: int = 50, _disp: bool = False,
    ):
        res = minimize(
            self.func, _init_params, method='L-BFGS-B',
            options={'maxiter': _max_iter, 'disp': _disp}
        )
        return res.x

    def fit(
            self, _arr: typing.Sequence[float],
            _max_iter: int = 200, _disp: bool = False
    ):
        assert len(_arr) > self.phi_num + self.alpha_num

        # init latent arr
        self.arr = np.array(_arr)
        self.y_arr = self.arr[self.phi_num:]
        if self.phi_num > 0:
            self.x_arr = stack_delay_arr(self.arr, self.phi_num)
        self.latent_arma_arr = np.zeros(
            len(self.arr) - self.phi_num + self.theta_num
        )
        self.latent_garch_arr = np.zeros(
            len(self.arr) - self.phi_num - self.alpha_num + self.beta_num
        )

        # 1. mu_arr
        if self.use_mu:
            self.mu_arr = np.mean(self.arr, keepdims=True)
        else:
            self.mu_arr = np.zeros(1)
        # 2. phi and theta
        if self.phi_arr is None:
            self.phi_arr = np.zeros(self.phi_num)
        if self.theta_arr is None:
            self.theta_arr = np.zeros(self.theta_num)
        # 3. logit_alpha_arr and logit_beta_arr
        init_value = 1. / np.sqrt(len(self.arr))
        if self.logit_alpha_arr is None:
            self.logit_alpha_arr = np.full(
                self.alpha_num, logit(init_value)
            )
        if self.logit_beta_arr is None:
            self.logit_beta_arr = np.full(
                self.beta_num, logit(init_value)
            )
        # 4. log_const_arr
        if self.log_const_arr is None:
            self.log_const_arr = np.log(np.var(self.arr, keepdims=True))

        init_params = np.concatenate((
            self.log_const_arr, self.phi_arr, self.theta_arr,
            self.logit_alpha_arr, self.logit_beta_arr
        ))
        if self.use_mu:
            init_params = np.concatenate((init_params, self.mu_arr))

        params = self.optimize(init_params, _max_iter, _disp)

        self.log_const_arr, self.phi_arr, self.theta_arr, \
        self.logit_alpha_arr, self.logit_beta_arr, _ = np.split(
            params, self.split_index
        )
        if self.use_mu:
            self.mu_arr = params[-1:]

    def predict(
            self,
            _arr: typing.Sequence[float],
            _latent_arma: typing.Sequence[float] = None,
            _latent_garch: typing.Sequence[float] = None,
    ) -> (float, float):
        # get data
        arr = np.array(_arr)
        if _latent_arma is None:
            latent_arma = self.latent_arma_arr
        else:
            latent_arma = np.array(_latent_arma)
        if _latent_garch is None:
            latent_garch = self.latent_garch_arr
        else:
            latent_garch = np.array(_latent_garch)

        # mean
        mean = self.mu_arr[0]
        if self.phi_num > 0:
            tmp_arr = arr[-self.phi_num:][::-1]
            mean += self.phi_arr.dot(tmp_arr)
        if self.theta_num > 0:
            tmp_arr = latent_arma[-self.theta_num:][::-1]
            mean += self.theta_arr.dot(tmp_arr)

        # var
        var = self.getConst()[0]
        tmp_arr = latent_arma[-self.alpha_num:][::-1]
        var += self.getAlphas().dot(tmp_arr ** 2)
        if self.beta_num > 0:
            tmp_arr = latent_garch[-self.beta_num:][::-1]
            var += self.getBetas().dot(tmp_arr)

        return mean, var

    def getPhis(self) -> np.ndarray:
        return self.phi_arr

    def getThetas(self) -> np.ndarray:
        return self.theta_arr

    def getAlphas(self) -> np.ndarray:
        return ilogit(self.logit_alpha_arr)

    def getBetas(self) -> np.ndarray:
        return ilogit(self.logit_beta_arr)

    def getConst(self) -> np.ndarray:
        return np.exp(self.log_const_arr)

    def getMu(self) -> np.ndarray:
        return self.mu_arr
