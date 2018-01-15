import typing

from TorchTSA.simulate.GARCHSim import GARCHSim


class ARCHSim(GARCHSim):

    def __init__(
            self,
            _alpha_arr: typing.Union[float, typing.Sequence[float]],
            _const: float, _mu: float = 0.0,
    ):
        if isinstance(_alpha_arr, float) or isinstance(_alpha_arr, int):
            _alpha_arr = (_alpha_arr,)

        super().__init__(
            _alpha_arr=_alpha_arr, _beta_arr=(),
            _const=_const, _mu=_mu
        )
