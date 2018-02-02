import typing

import numpy as np
from TorchTSA.utils.recursions import garch_recursion
from scipy.optimize import minimize

from TorchTSA.utils.math import ilogit, logit, logpdf
from TorchTSA.utils.op import stack_delay_arr


class GARCHModel:

    def __init__(
            self,
            _alpha_num: int = 1,
            _beta_num: int = 1,
            _use_mu: bool = True,
    ):
        assert _alpha_num > 0
        assert _beta_num >= 0

        # fitter params
        self.alpha_num = _alpha_num
        self.beta_num = _beta_num
        self.use_mu = _use_mu
        self.split_index = np.cumsum((
            1, self.alpha_num, self.beta_num
        ))

        # model params
        self.mu_arr: np.ndarray = None
        self.logit_alpha_arr: np.ndarray = None
        self.logit_beta_arr: np.ndarray = None
        self.log_const_arr: np.ndarray = None

        # buffers
        self.arr: np.ndarray = None
        self.y_arr: np.ndarray = None
        self.latent_arr: np.ndarray = None

    def func(self, _params: np.ndarray):
        # split params
        if self.use_mu:
            mu = _params[-1]
        else:
            mu = self.mu_arr
        const, alpha, beta, _ = np.split(_params, self.split_index)
        alpha = ilogit(alpha)
        beta = ilogit(beta)
        const = np.exp(const)

        square_arr = (self.arr - mu) ** 2
        arch_x_arr = stack_delay_arr(square_arr, self.alpha_num)
        if self.beta_num > 0:  # estimate latent_arr
            self.latent_arr[self.beta_num:] = alpha.dot(arch_x_arr) + const
            self.latent_arr[:self.beta_num] = square_arr.mean()
            garch_recursion(self.latent_arr, beta)
        else:
            self.latent_arr = alpha.dot(arch_x_arr) + const

        loss = -logpdf(
            self.y_arr, mu, self.latent_arr[self.beta_num:]
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
            _max_iter: int = 50, _disp: bool = False,
    ):
        assert len(_arr) > self.alpha_num

        # init target, latent_arr
        self.arr = np.array(_arr)
        self.y_arr = self.arr[self.alpha_num:]
        self.latent_arr = np.empty(
            len(self.arr) + self.beta_num - self.alpha_num
        )

        # 1. mu_arr
        if self.use_mu:
            self.mu_arr = np.mean(self.arr, keepdims=True)
        else:
            self.mu_arr = np.zeros(1)
        init_value = 1. / np.sqrt(len(self.arr))
        # 2. logit_alpha_arr
        if self.logit_alpha_arr is None:
            self.logit_alpha_arr = np.full(
                self.alpha_num, logit(init_value)
            )
        # 3. logit_beta_arr
        if self.logit_beta_arr is None:
            self.logit_beta_arr = np.full(
                self.beta_num, logit(init_value)
            )
        # 4. log_const_arr
        if self.log_const_arr is None:
            self.log_const_arr = np.log(np.var(self.arr, keepdims=True))

        init_params = np.concatenate((
            self.log_const_arr, self.logit_alpha_arr, self.logit_beta_arr
        ))
        if self.use_mu:
            init_params = np.concatenate((init_params, self.mu_arr))

        params = self.optimize(init_params, _max_iter, _disp)

        # update array
        self.log_const_arr, self.logit_alpha_arr, self.logit_beta_arr, _ = np.split(
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
        tmp_arr = (arr[-self.alpha_num:] - self.getMu()) ** 2
        tmp_arr = tmp_arr[::-1]
        value = self.getAlphas().dot(tmp_arr) + self.getConst()[0]
        if self.beta_num > 0:
            tmp_latent = latent[-self.beta_num:]
            tmp_latent = tmp_latent[::-1]
            value += self.getBetas().dot(tmp_latent)

        return value

    def getVolatility(self) -> typing.Union[None, np.ndarray]:
        if self.beta_num > 0:
            return np.sqrt(self.latent_arr[self.beta_num:])
        else:
            return None

    def getAlphas(self) -> np.ndarray:
        return ilogit(self.logit_alpha_arr)

    def getBetas(self) -> np.ndarray:
        return ilogit(self.logit_beta_arr)

    def getConst(self) -> np.ndarray:
        return np.exp(self.log_const_arr)

    def getMu(self) -> np.ndarray:
        return self.mu_arr
