import typing

from TorchTSA.simulate.ARMASim import ARMASim


class ARSim(ARMASim):

    def __init__(
            self,
            _phi_arr: typing.Union[float, typing.Sequence[float]],
            _const: float = 0.0, _sigma: float = 1.0,
    ):
        if isinstance(_phi_arr, float) or isinstance(_phi_arr, int):
            _phi_arr = (_phi_arr,)

        super().__init__(
            _phi_arr=_phi_arr, _const=_const, _sigma=_sigma
        )
