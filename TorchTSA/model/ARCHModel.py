from TorchTSA.model.GARCHModel import GARCHModel


class ARCHModel(GARCHModel):

    def __init__(
            self,
            _alpha_num: int = 1,
            _use_mu: bool = True,
    ):
        super().__init__(
            _alpha_num=_alpha_num,
            _beta_num=0,
            _use_mu=_use_mu
        )

    def getBetas(self):
        raise NotImplementedError('No Betas in ARCH model')
