import random
import typing

import numpy as np


class MASim:

    def __init__(
            self, _theta_arr: typing.Union[float, typing.Sequence[float]],
            _const: float = 0.0, _sigma: float = 1.0,
    ):
        if isinstance(_theta_arr, float) or isinstance(_theta_arr, int):
            _theta_arr = [_theta_arr]
        self.theta_arr = np.array(_theta_arr)
        self.theta_num = len(self.theta_arr)

        self.const = _const
        self.sigma = _sigma

        self.latent = [0.0] * self.theta_num
        self.ret = []

    def sample(self) -> float:
        tmp = self.latent[-self.theta_num:]
        tmp.reverse()
        latent_arr = np.array(tmp)
        new_value = np.sum(
            latent_arr * self.theta_arr
        )

        new_info = random.gauss(0, self.sigma)
        self.latent.append(new_info)

        new_value += new_info + self.const
        self.ret.append(new_value)

        return new_value

    def sample_n(self, _num: int) -> np.ndarray:
        for _ in range(_num):
            self.sample()
        return np.array(self.ret[-_num:])
