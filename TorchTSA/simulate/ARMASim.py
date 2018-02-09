import random
import typing

import numpy as np


class ARMASim:

    def __init__(
            self,
            _phi_arr: typing.Sequence[float] = (),
            _theta_arr: typing.Sequence[float] = (),
            _mu: float = 0.0, _sigma: float = 1.0,
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

        self.mu = _mu
        self.sigma = _sigma
        assert self.sigma > 0

        self.ret = [self.mu] * self.phi_num
        self.latent = [random.gauss(0, self.sigma) for _ in range(self.theta_num)]

    def sample(self) -> float:
        new_value = self.mu
        if self.phi_num > 0:  # AR part
            new_value += self.inv_phi_arr.dot(
                self.ret[-self.phi_num:]
            )
        if self.theta_num > 0:  # MA part
            new_value += self.inv_theta_arr.dot(
                self.latent[-self.theta_num:]
            )

        new_info = random.gauss(0, self.sigma)
        new_value += new_info
        self.ret.append(new_value)
        self.latent.append(new_info)

        return new_value

    def sample_n(self, _num: int) -> np.ndarray:
        for _ in range(_num):
            self.sample()
        return np.array(self.ret[-_num:])
