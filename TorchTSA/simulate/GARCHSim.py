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
        self.alpha_num = len(_alpha_arr)

        # MA part
        self.beta_arr = np.array(_beta_arr)
        assert np.all(self.beta_arr > 0)
        self.beta_num = len(_beta_arr)

        for a, b in zip(_alpha_arr, _beta_arr):
            assert a + b < 1

        self.const = _const
        assert self.const > 0

        self.mu = _mu

        self.ret = []
        self.square = [0.0] * self.alpha_num
        self.var = [0.0] * self.beta_num

    def sample(self) -> float:
        sigma2 = self.const
        if self.alpha_num > 0:  # AR part
            tmp = self.square[-self.alpha_num:]
            tmp.reverse()
            tmp = np.array(tmp)
            sigma2 += (tmp * self.alpha_arr).sum()
        if self.beta_num > 0:  # MA part
            tmp = self.var[-self.beta_num:]
            tmp.reverse()
            tmp = np.array(tmp)
            sigma2 += (tmp * self.beta_arr).sum()
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
