import random
import typing

import numpy as np


class ARMASim:

    def __init__(
            self,
            _phi_arr: typing.Sequence[float] = (),
            _theta_arr: typing.Sequence[float] = (),
            _const: float = 0.0, _sigma: float = 1.0,
    ):
        # AR part
        self.phi_arr = np.array(_phi_arr)
        self.phi_num = len(self.phi_arr)
        # MA part
        self.theta_arr = np.array(_theta_arr)
        self.theta_num = len(self.theta_arr)

        self.const = _const
        self.sigma = _sigma

        self.ret = [0.0] * self.phi_num
        self.latent = [0.0] * self.theta_num

    def sample(self) -> float:
        new_value = self.const
        if self.phi_num > 0:  # AR part
            tmp = self.ret[-self.phi_num:]
            tmp.reverse()
            tmp = np.array(tmp)
            new_value += (tmp * self.phi_arr).sum()
        if self.theta_num > 0:  # MA part
            tmp = self.latent[-self.theta_num:]
            tmp.reverse()
            tmp = np.array(tmp)
            new_value += (tmp * self.theta_arr).sum()

        new_info = random.gauss(0, self.sigma)
        new_value += new_info
        self.ret.append(new_value)
        self.latent.append(new_info)

        return new_value

    def sample_n(self, _num: int) -> np.ndarray:
        for _ in range(_num):
            self.sample()
        return np.array(self.ret[-_num:])
