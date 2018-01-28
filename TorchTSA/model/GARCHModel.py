import typing

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def logit(x):
    return np.log(x) - np.log(1 - x)


def ilogit(x):
    return 1. / (1. + np.exp(-x))


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

        # model params
        self.mu_arr: np.ndarray = None
        self.logit_alpha_arr: np.ndarray = None
        self.logit_beta_arr: np.ndarray = None
        self.log_const_arr: np.ndarray = None

        self.arr: np.ndarray = None
        self.y_arr: np.ndarray = None
        # latent for MA part
        self.latent_arr: np.ndarray = None

    @staticmethod
    def stack_delay_arr(
            _arr: typing.Sequence[float], _num: int
    ) -> np.ndarray:
        ret_list = []
        for i in range(_num):
            shift = i + 1
            ret_list.append(_arr[_num - shift: -shift])
        return np.stack(ret_list)

    def func(self, _params: np.ndarray):
        # split params
        if self.use_mu:
            mu = _params[-1]
        else:
            mu = self.mu_arr
        alpha = ilogit(_params[:self.alpha_num])
        beta = ilogit(_params[self.alpha_num:self.alpha_num + self.beta_num])
        const = np.exp(_params[self.alpha_num + self.beta_num])

        square_arr = (self.arr - mu) ** 2
        ar_x_arr = self.stack_delay_arr(square_arr, self.alpha_num)
        if self.beta_num > 0:  # estimate latent_arr
            self.latent_arr[self.beta_num:] = alpha.dot(ar_x_arr) + const
            self.latent_arr[:self.beta_num] = max(0, mu / (1 - beta.sum()))
            ma_x_arr = self.stack_delay_arr(self.latent_arr, self.beta_num)
            self.latent_arr[self.beta_num:] += beta.dot(ma_x_arr)
            ma_x_arr = self.stack_delay_arr(self.latent_arr, self.beta_num)

        var = alpha.dot(ar_x_arr)
        if self.beta_num > 0:
            var = var + beta.dot(ma_x_arr)
        var = var + const
        loss = -norm(
            loc=mu, scale=np.sqrt(var)
        ).logpdf(self.y_arr).mean()

        return loss

    def fit(
            self, _arr: typing.Sequence[float],
            _max_iter: int = 50, _disp: bool = False,
    ):
        assert len(_arr) > self.alpha_num

        # init params and latent
        self.arr = np.array(_arr)

        # get vars and optimizer
        # 1. mu_arr
        if self.use_mu:
            self.mu_arr = np.mean(self.arr, keepdims=True)
        else:
            self.mu_arr = np.zeros(1)

        init_value = 1. / np.sqrt(len(self.arr))
        # 2. logit_alpha_arr
        if self.logit_alpha_arr is None:
            self.logit_alpha_arr = np.empty(self.alpha_num)
            self.logit_alpha_arr.fill(logit(init_value))
        # 3. logit_beta_arr
        if self.beta_num > 0:
            if self.logit_beta_arr is None:
                self.logit_beta_arr = np.empty(self.beta_num)
                self.logit_beta_arr.fill(logit(init_value))
        # 4. log_const_arr
        if self.log_const_arr is None:
            self.log_const_arr = np.empty(1)
            self.log_const_arr.fill(np.log(init_value))

        # init target, latent_arr
        self.y_arr = self.arr[self.alpha_num:]
        if self.beta_num > 0:  # alloc latent_arr
            self.latent_arr = np.empty(
                len(self.arr) + self.beta_num - self.alpha_num
            )

        init_params = self.logit_alpha_arr
        if self.beta_num > 0:
            init_params = np.concatenate((init_params, self.logit_beta_arr))
        init_params = np.concatenate((init_params, self.log_const_arr))
        if self.use_mu:
            init_params = np.concatenate((init_params, self.mu_arr))

        res = minimize(
            self.func, init_params, method='L-BFGS-B',
            options={'maxiter': _max_iter, 'disp': _disp}
        )
        params = res.x

        self.logit_alpha_arr = params[:self.alpha_num]
        self.logit_beta_arr = params[self.alpha_num:self.alpha_num + self.beta_num]
        self.log_const_arr = params[self.alpha_num + self.beta_num]
        if self.use_mu:
            self.mu_arr = params[-1]

    def getAlphas(self) -> np.ndarray:
        return ilogit(self.logit_alpha_arr)

    def getBetas(self) -> np.ndarray:
        return ilogit(self.logit_beta_arr)

    def getConst(self) -> np.ndarray:
        return np.exp(self.log_const_arr)

    def getMu(self) -> np.ndarray:
        return self.mu_arr
