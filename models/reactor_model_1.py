"""Educational point-kinetics/two-node PWR core model. NOT for plant operations.

Parameters preserve the earlier Model 1 development assumptions. Rahim et al.
(2026), DOI 10.1016/j.nxener.2026.100597, Table 2 kinetics coefficients
are provisional (especially the printed lambda_4). Thermal model is our own.
"""
from dataclasses import dataclass
import numpy as np

PCM = 1e-5

@dataclass(frozen=True)
class ReactorParameters:
    q_rated: float = 3.40e9
    t_in: float = 553.85
    mdot: float = 14275.6
    cp_c: float = 5590.0
    mc: float = 11665.8
    mf: float = 96241.0
    cp_f: float = 308.0
    tf_ref: float = 900.0
    generation_time: float = 3e-5
    alpha_f: float = -2.16e-5
    alpha_c: float = -1.80e-4

    @property
    def f(self): return self.mdot*self.cp_c
    @property
    def mc1(self): return self.mc/2
    @property
    def mc2(self): return self.mc/2
    @property
    def tout_ref(self): return self.t_in+self.q_rated/self.f
    @property
    def r(self): return np.sqrt((self.tf_ref-self.t_in)/(self.tf_ref-self.tout_ref))-1
    @property
    def ua(self): return self.f*self.r
    @property
    def tc1_ref(self): return (self.t_in+self.r*self.tf_ref)/(1+self.r)
    @property
    def tc2_ref(self): return self.tout_ref

DECAY = np.array([0.012437, 0.03050, 0.11141, 0.03013, 1.12866, 3.0130])
BETA_I = np.array([2.15e-4, 1.424e-3, 1.274e-3, 2.568e-3, 7.48e-4, 2.73e-4])
BETA = BETA_I.sum()

def initial_core_state(p: ReactorParameters) -> np.ndarray:
    return np.array([1., *([1.]*6), p.tf_ref, p.tc1_ref, p.tc2_ref])

def rho_thermal(core, p: ReactorParameters):
    tf, tc1, tc2=core[7:10]
    return p.alpha_f*(tf-p.tf_ref)+p.alpha_c*((tc1-p.tc1_ref)+(tc2-p.tc2_ref))

def core_rhs(core, applied_rho, inlet_t, flow_multiplier, p: ReactorParameters):
    pn=core[0]; c=core[1:7]; tf,tc1,tc2=core[7:10]
    rho=applied_rho+rho_thermal(core,p)
    dp=((rho-BETA)*pn+np.dot(BETA_I,c))/p.generation_time
    dc=DECAY*(pn-c)
    q1=p.ua*(tf-tc1); q2=p.ua*(tf-tc2)
    f=p.f*flow_multiplier
    dtf=(p.q_rated*pn-q1-q2)/(p.mf*p.cp_f)
    dtc1=(q1+f*(inlet_t-tc1))/(p.mc1*p.cp_c)
    dtc2=(q2+f*(tc1-tc2))/(p.mc2*p.cp_c)
    return np.r_[dp,dc,dtf,dtc1,dtc2]
