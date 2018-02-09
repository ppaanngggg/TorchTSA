import logging

import matplotlib.pyplot as plt
import pyflux as pf

from TorchTSA.model import MAModel
from TorchTSA.simulate import MASim

logging.basicConfig(level=logging.INFO)

# simulate data
ma_sim = MASim(_theta_arr=[0.3, -0.2], _mu=0.0, _sigma=0.5)
sim_data = ma_sim.sample_n(3000)

ma_model = MAModel(2, _use_mu=True)
ma_model.fit(sim_data)
print(ma_model.getThetas(), ma_model.getMu(), ma_model.getSigma())
print(ma_model.predict(sim_data))

plt.plot(ma_sim.latent)
plt.plot(ma_model.latent_arr)
plt.show()

pf_model = pf.ARIMA(data=sim_data, ar=0, ma=2, integ=0)
pf_ret = pf_model.fit("MLE")
pf_ret.summary()
print(pf_model.predict())
