"""Educational Model 1: inlet-temperature disturbance rejection.

Requires model_1_open_loop.py, model_1_linear_validation.py, and
model_1_full_model_PI.py in this directory. Does not change those files.
Not a validated AP1000 or licensed control system.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model_1_open_loop as m
from model_1_linear_validation import build_matrices
from model_1_full_model_PI import TAU_V, TAU_M, PCM_TO_RHO, closed_loop_matrices

K_PROP, K_INT = 30.0, 2.0  # provisional full-model PI gains, pcm/K and pcm/(K s)
T_STEP = 50.0
DELTA_TIN = -0.1  # K
T_END = 500.0


def core_rhs(x, rho_rod, inlet):
    pn, c = x[0], x[1:7]
    tf, tc1, tc2 = x[7:10]
    rho = m.total_reactivity(tf, tc1, tc2, rho_rod)
    dp = ((rho - m.BETA)*pn + m.BETA_I@c)/m.LAMBDA
    dc = m.DECAY*(pn-c)
    q1 = m.U*(tf-tc1)
    q2 = m.U*(tf-tc2)
    dtf = (m.Q_RATED*pn-q1-q2)/(m.M_F*m.CP_F)
    dtc1 = (q1+m.F*(inlet-tc1))/(m.M_C1*m.CP_C)
    dtc2 = (q2+m.F*(tc1-tc2))/(m.M_C2*m.CP_C)
    return np.r_[dp,dc,dtf,dtc1,dtc2]


def simulate_piecewise(rhs, initial):
    """Integrate in two segments so solver sees exact disturbance onset."""
    grids = [np.linspace(0,T_STEP,301),np.linspace(T_STEP,T_END,1801)]
    parts = []
    state = np.asarray(initial,dtype=float)
    for grid, disturbance in zip(grids,[0.0,DELTA_TIN]):
        sol = solve_ivp(lambda t,y: rhs(t,y,disturbance),
                        (grid[0],grid[-1]),state,t_eval=grid,method='Radau',
                        rtol=1e-8,atol=1e-10,max_step=1.0)
        if not sol.success: raise RuntimeError(sol.message)
        parts.append(sol)
        state=sol.y[:,-1]
    t=np.r_[parts[0].t,parts[1].t[1:]]
    x=np.column_stack([parts[0].y,parts[1].y[:,1:]])
    return t,x


def pi_command(y):
    # Temperature setpoint is held constant at the rated reference.
    error = m.TC2_REF-y[11]
    return K_PROP*error+K_INT*y[12]


def main():
    A,B,C,D=build_matrices()
    Bd=np.zeros(10)
    Bd[8]=m.F/(m.M_C1*m.CP_C) # = mass flow / coolant inventory [1/s]
    Gd_ss=(A,Bd.reshape(-1,1),C,D)

    # Three independent paths: open-loop nonlinear; PI-controlled nonlinear;
    # PI-controlled linearized 13-state closed loop with explicit disturbance input.
    t,x_open=simulate_piecewise(lambda t,x,d:core_rhs(x,0.0,m.T_IN+d),m.Y0)

    y0=np.r_[m.Y0,0.0,m.TC2_REF,0.0]
    def nonlinear_closed(t,y,d):
        cmd_pcm=pi_command(y)
        applied=y[10]
        return np.r_[core_rhs(y[:10],applied,m.T_IN+d),
                     (PCM_TO_RHO*cmd_pcm-applied)/TAU_V,
                     (y[9]-y[11])/TAU_M,
                     m.TC2_REF-y[11]]
    tc,y_cl=simulate_piecewise(nonlinear_closed,y0)

    Acl,Bsp=closed_loop_matrices(K_PROP,K_INT)
    Bdist=np.zeros(13)
    Bdist[:10]=Bd
    tl,x_lin=simulate_piecewise(lambda t,x,d:Acl@x+Bdist*d,np.zeros(13))
    assert np.allclose(t,tc) and np.allclose(t,tl)
    assert np.max(np.linalg.eigvals(Acl).real)<0
    assert np.max(np.abs(core_rhs(m.Y0,0.0,m.T_IN)))<1e-8

    dTin=np.where(t<T_STEP,0.0,DELTA_TIN)
    out_open=x_open[9]-m.TC2_REF
    out_closed=y_cl[11]-m.TC2_REF
    out_lin=x_lin[11]
    err=out_closed-out_lin
    cmd=np.array([pi_command(y_cl[:,j]) for j in range(y_cl.shape[1])])
    applied=y_cl[10]/PCM_TO_RHO
    rho_thermal=(m.ALPHA_F*(y_cl[7]-m.TF_REF)+
                 m.ALPHA_C*((y_cl[8]-m.TC1_REF)+(y_cl[9]-m.TC2_REF)))/PCM_TO_RHO
    rho_total=applied+rho_thermal
    selected=t>=T_STEP
    duration=t[selected]-T_STEP
    pe_open=np.max(np.abs(out_open[selected]))
    pe_closed=np.max(np.abs(out_closed[selected]))
    iae_open=np.trapezoid(np.abs(out_open[selected]),duration)
    iae_closed=np.trapezoid(np.abs(out_closed[selected]),duration)
    print('MODEL 1 INLET TEMPERATURE DISTURBANCE — EDUCATIONAL BENCHMARK')
    print(f'Input: {DELTA_TIN:+g} K at t={T_STEP:g} s; setpoint held at {m.TC2_REF:.4f} K')
    print(f'PI: Kp={K_PROP:g} pcm/K, Ki={K_INT:g} pcm/(K s)')
    print(f'Bdist[8] = {Bd[8]:.8f} 1/s; Gd(s) = C(sI-A)^(-1) Bdist')
    print(f'Linear closed-loop max real pole: {max(np.linalg.eigvals(Acl).real):.6g} /s')
    print(f'Peak |measured/outlet ΔT| open loop: {pe_open:.7f} K')
    print(f'Peak |measured ΔT| closed loop:    {pe_closed:.7f} K')
    print(f'IAE open loop:   {iae_open:.7f} K*s')
    print(f'IAE closed loop: {iae_closed:.7f} K*s')
    print(f'Final open-loop outlet deviation: {out_open[-1]:+.7f} K')
    print(f'Final closed-loop sensor deviation: {out_closed[-1]:+.7f} K')
    print(f'Linear/nonlinear closed-loop max error: {np.max(np.abs(err)):.3e} K')
    print(f'Linear/nonlinear closed-loop RMSE: {np.sqrt(np.mean(err**2)):.3e} K')
    print(f'Peak |rod command| = {np.max(np.abs(cmd)):.6f} pcm')
    print('No saturation, noise or real rod kinematics; benchmark values only.')

    fig,ax=plt.subplots(4,1,figsize=(11,12),sharex=True,constrained_layout=True)
    ax[0].plot(t,dTin,label='Inlet temperature disturbance')
    ax[0].set_ylabel('ΔTin [K]');ax[0].legend()
    ax[1].plot(t,out_open,label='Open-loop nonlinear outlet',lw=1.8)
    ax[1].plot(t,out_closed,label='PI-controlled nonlinear sensor',lw=2)
    ax[1].plot(t,out_lin,'--',label='PI-controlled linear prediction',lw=1.2)
    ax[1].axhline(0,ls=':',color='gray',label='Held setpoint')
    ax[1].set_ylabel('Δ outlet / measured T [K]');ax[1].legend()
    ax[2].plot(t,cmd,label='Commanded rod reactivity')
    ax[2].plot(t,applied,label='Applied rod reactivity')
    ax[2].set_ylabel('Reactivity [pcm]');ax[2].legend()
    ax[3].plot(t,y_cl[0]-1,label='Closed-loop power deviation [p.u.]')
    ax[3].plot(t,x_open[0]-1,label='Open-loop power deviation [p.u.]')
    ax[3].set_ylabel('Δ reactor power [p.u.]');ax[3].set_xlabel('Time [s]');ax[3].legend()
    for a in ax:
        a.axvline(T_STEP,ls=':',color='gray');a.grid(alpha=.25)
    fig.suptitle('Model 1: Inlet-temperature disturbance rejection (educational benchmark)')
    path=Path(__file__).with_name('model_1_disturbance_rejection.png')
    fig.savefig(path,dpi=155)
    print('Saved plot:',path)

    fig2,ax2=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    ax2[0].plot(t,applied,label='Applied rod reactivity')
    ax2[0].plot(t,rho_thermal,label='Temperature reactivity feedback')
    ax2[0].plot(t,rho_total,label='Total reactivity')
    ax2[0].set_ylabel('Reactivity [pcm]');ax2[0].legend()
    ax2[1].plot(t,err,label='Nonlinear − linear measured outlet')
    ax2[1].set_ylabel('Validation difference [K]');ax2[1].set_xlabel('Time [s]');ax2[1].legend()
    for a in ax2: a.axvline(T_STEP,ls=':',color='gray');a.grid(alpha=.25)
    fig2.savefig(Path(__file__).with_name('model_1_disturbance_diagnostics.png'),dpi=155)

if __name__=='__main__': main()
