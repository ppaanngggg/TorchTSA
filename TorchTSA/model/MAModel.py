import numpy as np

from TorchTSA.model.ARMAModel import ARMAModel


class MAModel(ARMAModel):

    def __init__(
            self,
            _theta_num: int = 1,
            _use_const: bool = True
    ):
        super().__init__(
            _phi_num=0,
            _theta_num=_theta_num,
            _use_const=_use_const,
        )

    def getPhis(self):
        raise NotImplementedError('No Phi in MA Model')