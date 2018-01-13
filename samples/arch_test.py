import logging

import pyflux as pf

from TorchTSA.model import ARCHModel
from TorchTSA.simulate import ARCHSim

logging.basicConfig(level=logging.INFO)

arch_sim = ARCHSim((0.6, 0.1), _const=0.1, _mu=0.0)
sim_data = arch_sim.sample_n(1000)

arch_model = ARCHModel(2, _use_mu=False)
arch_model.fit(sim_data)
print(
    arch_model.getAlpha(),
    arch_model.getConst(),
    arch_model.getMu(),
)

pf_model = pf.GARCH(sim_data, 0, 2)
pf_ret = pf_model.fit("MLE")
pf_ret.summary()
