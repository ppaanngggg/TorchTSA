import logging

import pyflux as pf

from TorchTSA.model import MAModel
from TorchTSA.simulate import MASim

logging.basicConfig(level=logging.INFO)

# simulate data
ma_sim = MASim(_theta_arr=[0.3, -0.2], _const=0.1)
sim_data = ma_sim.sample_n(1000)

ma_model = MAModel(2, _use_const=True)
ma_model.fit(sim_data)
print(ma_model.getThetas(), ma_model.getConst())
print(ma_model.sigma_arr)

pf_model = pf.ARIMA(data=sim_data, ar=0, ma=2, integ=0)
pf_ret = pf_model.fit("MLE")
pf_ret.summary()
