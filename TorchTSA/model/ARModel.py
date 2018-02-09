from TorchTSA.model.ARMAModel import ARMAModel


class ARModel(ARMAModel):

    def __init__(
            self, _phi_num: int = 1,
            _use_mu: bool = True
    ):
        super().__init__(
            _phi_num=_phi_num,
            _theta_num=0,
            _use_mu=_use_mu
        )

    def getThetas(self):
        raise NotImplementedError('No Theta in AR Model')
