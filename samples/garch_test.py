import pyflux as pf

from TorchTSA.simulate import GARCHSim

garch_sim = GARCHSim(
    (0.5,), (0.3, 0.1), _const=0.1
)
sim_data = garch_sim.sample_n(2000)

pf_model = pf.GARCH(sim_data, 2, 1)
pf_ret = pf_model.fit("MLE")
pf_ret.summary()
