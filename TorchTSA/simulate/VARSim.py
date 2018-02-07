import numpy as np


class VARSim:

    def __init__(
            self, _phi_arr: np.ndarray,
            _mu_arr: np.ndarray, _cov_arr: np.ndarray,
    ):
        self.phi_arr = _phi_arr
        self.inv_phi_arr = self.phi_arr[::-1]
        self.phi_num = len(self.phi_arr)
        self.mu_arr = _mu_arr
        self.cov_arr = _cov_arr

        self.ret = [self.mu_arr] * self.phi_num

    def sample(self) -> np.ndarray:
        new_value = np.random.multivariate_normal(
            mean=self.mu_arr, cov=self.cov_arr
        )
        new_value += np.matmul(self.inv_phi_arr, np.expand_dims(
            self.ret[-self.phi_num:], -1
        )).sum(0).squeeze()
        self.ret.append(new_value)
        return new_value

    def sample_n(self, _num: int) -> np.ndarray:
        for _ in range(_num):
            self.sample()
        return np.array(self.ret[-_num:])
