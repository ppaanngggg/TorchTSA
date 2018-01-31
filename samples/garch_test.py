import logging
import time

import pyflux as pf

from TorchTSA.model import GARCHModel
from TorchTSA.model import IGARCHModel
from TorchTSA.simulate import GARCHSim

logging.basicConfig(level=logging.INFO)

garch_sim = GARCHSim(
    (0.5,), (0.5,), _const=0.1
)
sim_data = garch_sim.sample_n(2000)

# garch model
garch_model = GARCHModel(1, 1, _use_mu=True)
start_time = time.time()
garch_model.fit(sim_data)
print('fitting time:', time.time() - start_time)
print(
    garch_model.getAlphas(), garch_model.getBetas(),
    garch_model.getConst(), garch_model.getMu(),
)
print('predict value:', garch_model.predict(sim_data))

# igarch model
igarch_model = IGARCHModel(1, 1, _use_mu=True)
start_time = time.time()
igarch_model.fit(sim_data)
print('fitting time:', time.time() - start_time)
print(
    igarch_model.getAlphas(), igarch_model.getBetas(),
    igarch_model.getConst(), igarch_model.getMu(),
)
print('predict value:', igarch_model.predict(sim_data))

# pyflux's garch model
pf_model = pf.GARCH(sim_data, 1, 1)
start_time = time.time()
pf_ret = pf_model.fit("MLE")
print('fitting time:', time.time() - start_time)
pf_ret.summary()
print(pf_model.predict())
