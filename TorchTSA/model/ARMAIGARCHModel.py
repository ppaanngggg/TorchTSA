import numpy as np
from scipy.optimize import minimize

from TorchTSA.model.ARMAGARCHModel import ARMAGARCHModel
from TorchTSA.utils.math import ilogit


class ARMAIGARCHModel(ARMAGARCHModel):
    def optimize(
            self, _init_params: np.ndarray,
            _max_iter: int = 200, _disp: bool = False,
    ):
        cons = {
            'type': 'eq',
            'fun': lambda x: ilogit(
                x[self.split_index[2]:self.split_index[4]]
            ).sum() - 1.0,
        }
        ret = minimize(
            self.func, _init_params, constraints=cons,
            method='SLSQP', options={'maxiter': _max_iter, 'disp': _disp}
        )
        return ret.x
