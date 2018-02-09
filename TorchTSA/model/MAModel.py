import numpy as np

from TorchTSA.model.ARMAModel import ARMAModel


class MAModel(ARMAModel):

    def __init__(
            self,
            _theta_num: int = 1,
            _use_mu: bool = True
    ):
        super().__init__(
            _phi_num=0,
            _theta_num=_theta_num,
            _use_mu=_use_mu,
        )

    def getPhis(self):
        raise NotImplementedError('No Phi in MA Model')