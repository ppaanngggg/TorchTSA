import random
import typing

import numpy as np


class ARSim:
    def __init__(
            self, _theta_arr: typing.Union[float, typing.Sequence[float]],
            _const: float = 0.0, _sigma: float = 1.0,
    ):
        if isinstance(_theta_arr, float) or isinstance(_theta_arr, int):
            _theta_arr = [_theta_arr]
        self.theta_arr = np.array(_theta_arr)
        self.const = _const
        self.sigma = _sigma

        self.ret = [self.const] * len(self.theta_arr)

    def sample(self) -> float:
        value_arr = np.array(self.ret[-len(self.theta_arr):])
        new_value = (value_arr * self.theta_arr).sum() + random.gauss(0, self.sigma)
        self.ret.append(new_value)

        return new_value

    def sample_n(self, _num: int) -> typing.List[float]:
        for _ in range(_num):
            self.sample()
        return self.ret[-_num:]
