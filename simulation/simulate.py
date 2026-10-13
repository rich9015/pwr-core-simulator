"""Reusable 13-state nonlinear simulator: 10 core + actuator + sensor + PI integral.
One discontinuity is integrated exactly using two solve_ivp segments.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from models.reactor_model_1 import (ReactorParameters, PCM, initial_core_state,
                                    core_rhs, rho_thermal)
from models.controller import ControlParameters
from models.disturbances import Scenario

@dataclass
class Result:
    frame: pd.DataFrame
    metrics: dict


def simulate(scenario: Scenario, p=None, control=None) -> Result:
    p=p or ReactorParameters()
    control=control or ControlParameters()
    if not (scenario.duration > scenario.step_time > 0):
        raise ValueError('Require 0 < step time < duration.')
    if 1+scenario.flow_step_percent/100<=0:
        raise ValueError('Coolant flow must remain positive.')
    if control.tau_m <=0 or control.tau_v<=0:
        raise ValueError('Actuator and sensor time constants must be positive.')
    y0=np.r_[initial_core_state(p),0.0,p.tout_ref,0.0]
    def rhs(t,y,set_k,inlet_k,flow,rod_pcm):
        err=(p.tout_ref+set_k)-y[11]
        cmd_pcm=(control.kp*err+control.ki*y[12]) if scenario.closed_loop else rod_pcm
        core=core_rhs(y[:10],y[10],p.t_in+inlet_k,flow,p)
        return np.r_[core,(PCM*cmd_pcm-y[10])/control.tau_v,
                     (y[9]-y[11])/control.tau_m,
                     err if scenario.closed_loop else 0.0]
    # Exact step onset, no forcing the solver to infer discontinuity.
    grids=(np.linspace(0,scenario.step_time,max(40,int(scenario.step_time*3)+1)),
           np.linspace(scenario.step_time,scenario.duration,max(140,int((scenario.duration-scenario.step_time)*3)+1)))
    parts=[]; state=y0
    for j,grid in enumerate(grids):
        vals=scenario.values(grid[0] if j==0 else scenario.step_time+1e-8)
        if j==0: vals=(0.,0.,1.,0.)
        sol=solve_ivp(lambda t,y:rhs(t,y,*vals),(grid[0],grid[-1]),state,
                      t_eval=grid,method='Radau',rtol=2e-8,atol=2e-10,max_step=2.0)
        if not sol.success: raise RuntimeError(sol.message)
        parts.append(sol);state=sol.y[:,-1]
    t=np.r_[parts[0].t,parts[1].t[1:]]
    y=np.column_stack((parts[0].y,parts[1].y[:,1:]))
    active=t>=scenario.step_time
    sp=p.tout_ref+np.where(active,scenario.setpoint_step_k,0.)
    tin=p.t_in+np.where(active,scenario.inlet_step_k,0.)
    flow=1+np.where(active,scenario.flow_step_percent/100.,0.)
    err=sp-y[11]
    cmd=(control.kp*err+control.ki*y[12]) if scenario.closed_loop else np.where(active,scenario.rod_step_pcm,0.)
    applied=y[10]/PCM
    thermal=p.alpha_f*(y[7]-p.tf_ref)/PCM+p.alpha_c*((y[8]-p.tc1_ref)+(y[9]-p.tc2_ref))/PCM
    frame=pd.DataFrame(dict(time_s=t,setpoint_k=sp,tmeas_k=y[11],tout_k=y[9],
        tf_k=y[7],tc1_k=y[8],pn=y[0],rod_command_pcm=cmd,
        rod_applied_pcm=applied,thermal_feedback_pcm=thermal,
        total_reactivity_pcm=applied+thermal, inlet_k=tin,flow_fraction=flow,
        error_k=err))
    mask=active
    errs=err[mask]
    ts=t[mask]
    dev = y[11, mask] - p.tout_ref
    metrics={'peak_measured_deviation_k': float(np.max(np.abs(dev))),
             'integrated_measured_deviation_k_s': float(np.trapezoid(np.abs(dev), ts)),
             'final_measured_deviation_k': float(y[11, -1] - p.tout_ref),'peak_absolute_error_k':float(np.max(np.abs(errs))),
             'iae_k_s':float(np.trapezoid(np.abs(errs),ts)),
             'final_error_k':float(err[-1]),
             'peak_rod_command_pcm':float(np.max(np.abs(cmd))),
             'final_power_pu':float(y[0,-1])}
    return Result(frame,metrics)
