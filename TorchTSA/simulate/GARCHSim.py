import math
import random
import typing

import numpy as np


class GARCHSim:

    def __init__(
            self,
            _alpha_arr: typing.Sequence[float],
            _beta_arr: typing.Sequence[float],
            _const: float,
            _mu: float = 0.0,
    ):
        # AR part
        self.alpha_arr = np.array(_alpha_arr)
        assert np.all(self.alpha_arr > 0)
        self.inv_alpha_arr = self.alpha_arr[::-1]
        self.alpha_num = len(_alpha_arr)

        # MA part
        self.beta_arr = np.array(_beta_arr)
        assert np.all(self.beta_arr > 0)
        self.inv_beta_arr = self.beta_arr[::-1]
        self.beta_num = len(_beta_arr)

        # safety assert
        params_sum = self.alpha_arr.sum() + self.beta_arr.sum()
        assert params_sum <= 1.0

        self.const = _const
        assert self.const > 0

        self.mu = _mu

        self.ret = []
        init_value = self.const * (1.0 + params_sum)
        self.square = [init_value] * self.alpha_num
        self.var = [init_value] * self.beta_num

    def sample(self) -> float:
        sigma2 = self.const
        if self.alpha_num > 0:  # AR part
            sigma2 += self.inv_alpha_arr.dot(
                self.square[-self.alpha_num:]
            )
        if self.beta_num > 0:  # MA part
            sigma2 += self.inv_beta_arr.dot(
                self.var[-self.beta_num:]
            )
        # update value
        new_info = random.gauss(0, math.sqrt(sigma2))
        new_value = self.mu + new_info
        # buf all
        self.ret.append(new_value)
        self.square.append(new_info ** 2)
        self.var.append(sigma2)

        return new_value

    def sample_n(self, _num: int) -> np.ndarray:
        for _ in range(_num):
            self.sample()
        return np.array(self.ret[-_num:])
