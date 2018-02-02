import math
import random
import typing

import numpy as np


class ARMAGARCHSim:
    def __init__(
            self,
            _phi_arr: typing.Sequence[float],
            _theta_arr: typing.Sequence[float],
            _alpha_arr: typing.Sequence[float],
            _beta_arr: typing.Sequence[float],
            _const: float,
            _mu: float = 0.0,
    ):
        # AR part
        self.phi_arr = np.array(_phi_arr)
        assert np.all(np.abs(self.phi_arr) < 1)
        self.inv_phi_arr = self.phi_arr[::-1]
        self.phi_num = len(self.phi_arr)
        # MA part
        self.theta_arr = np.array(_theta_arr)
        assert np.all(np.abs(self.theta_arr) < 1)
        self.inv_theta_arr = self.theta_arr[::-1]
        self.theta_num = len(self.theta_arr)
        # ARCH part
        self.alpha_arr = np.array(_alpha_arr)
        assert np.all(self.alpha_arr > 0)
        self.inv_alpha_arr = self.alpha_arr[::-1]
        self.alpha_num = len(_alpha_arr)
        # G part
        self.beta_arr = np.array(_beta_arr)
        assert np.all(self.beta_arr > 0)
        self.inv_beta_arr = self.beta_arr[::-1]
        self.beta_num = len(_beta_arr)

        # safety assert
        params_sum = self.alpha_arr.sum() + self.beta_arr.sum()
        assert params_sum <= 1.0

        # const and mu
        self.const = _const
        assert self.const > 0
        self.mu = _mu

        # buffers
        self.ret = [self.mu] * self.phi_num
        init_value = self.const * (1.0 + params_sum)
        self.new_buf = [0.0] * self.theta_num
        self.square_buf = [init_value] * self.alpha_num
        self.var_buf = [init_value] * self.beta_num

    def sample(self) -> float:
        sigma2 = self.const
        if self.alpha_num > 0:  # ARCH part
            sigma2 += self.inv_alpha_arr.dot(
                self.square_buf[-self.alpha_num:]
            )
        if self.beta_num > 0:  # G part
            sigma2 += self.inv_beta_arr.dot(
                self.var_buf[-self.beta_num:]
            )
        # update value
        new_info = random.gauss(0, math.sqrt(sigma2))
        new_value = self.mu + new_info
        if self.phi_num > 0:  # AR part
            new_value += self.inv_phi_arr.dot(
                self.ret[-self.phi_num:]
            )
        if self.theta_num > 0:  # MA part
            new_value += self.inv_theta_arr.dot(
                self.new_buf[-self.theta_num:]
            )
        # buf all
        self.ret.append(new_value)
        self.new_buf.append(new_info)
        self.square_buf.append(new_info ** 2)
        self.var_buf.append(sigma2)

        return new_value

    def sample_n(self, _num: int) -> np.ndarray:
        for _ in range(_num):
            self.sample()
        return np.array(self.ret[-_num:])
