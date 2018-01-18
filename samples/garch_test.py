import logging

import pyflux as pf

from TorchTSA.model import GARCHModel
from TorchTSA.simulate import GARCHSim

logging.basicConfig(level=logging.INFO)

garch_sim = GARCHSim(
    (0.5,), (0.2,), _const=0.1
)
sim_data = garch_sim.sample_n(2000)

garch_model = GARCHModel(1, 1)
garch_model.fit(sim_data, _max_iter=50)
print(
    garch_model.getAlphas(), garch_model.getBetas(),
    garch_model.getConst(), garch_model.getMu(),
)

pf_model = pf.GARCH(sim_data, 1, 1)
pf_ret = pf_model.fit("MLE")
pf_ret.summary()
