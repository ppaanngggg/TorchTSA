import time

from TorchTSA.model import ARMAGARCHModel
from TorchTSA.model import ARMAIGARCHModel
from TorchTSA.simulate import ARMAGARCHSim

sim = ARMAGARCHSim(
    _phi_arr=(0.8,), _theta_arr=(-0.5,),
    _alpha_arr=(0.15,), _beta_arr=(0.8,),
    _const=0.01
)
sim_data = sim.sample_n(2000)

arma_garch_model = ARMAGARCHModel()
start_time = time.time()
arma_garch_model.fit(sim_data)
print(time.time() - start_time)
print(
    arma_garch_model.getPhis(), arma_garch_model.getThetas(),
    arma_garch_model.getAlphas(), arma_garch_model.getBetas(),
    arma_garch_model.getConst(), arma_garch_model.getMu(),
)

arma_igarch_model = ARMAIGARCHModel()
start_time = time.time()
arma_igarch_model.fit(sim_data)
print(time.time() - start_time)
print(
    arma_igarch_model.getPhis(), arma_igarch_model.getThetas(),
    arma_igarch_model.getAlphas(), arma_igarch_model.getBetas(),
    arma_igarch_model.getConst(), arma_igarch_model.getMu(),
)