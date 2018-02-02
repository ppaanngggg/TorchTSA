import numpy as np
from scipy.optimize import minimize

from TorchTSA.model import GARCHModel
from TorchTSA.utils.math import ilogit


class IGARCHModel(GARCHModel):
    def optimize(
            self, _init_params: np.ndarray,
            _max_iter: int = 50, _disp: bool = False,
    ):
        cons = {
            'type': 'eq',
            'fun': lambda x: ilogit(
                x[1:self.alpha_num + self.beta_num + 1]
            ).sum() - 1.0,
        }
        ret = minimize(
            self.func, _init_params, constraints=cons,
            method='SLSQP', options={'maxiter': _max_iter, 'disp': _disp}
        )
        return ret.x
