"""Educational Model 1 PI design against the FULL 10-state linearized reactor.

Requires model_1_open_loop.py, model_1_linear_validation.py.
Rod-actuator: 2 s lag; sensor: 1 s lag; reactivity command in pcm.
This is an illustrative controls study, NOT reactor operating guidance.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
import model_1_open_loop as m
from model_1_linear_validation import build_matrices

TAU_V=2.0
TAU_M=1.0
PCM_TO_RHO=1e-5
STEP_TIME=20.0
STEP_K=0.01  # modest temp setpoint change for small-signal validation
T_END=500.0


def augmented_plant():
    A, B, C, D=build_matrices()
    # x=[Pn,C1..C6,Tf,Tc1,Tc2,applied_rho[dimensionless],Tmeas_delta[K]]
    Aa=np.zeros((12,12))
    Aa[:10,:10]=A
    Aa[:10,10]=B[:,0]
    Aa[10,10]=-1/TAU_V
    Aa[11,9]=1/TAU_M
    Aa[11,11]=-1/TAU_M
    Bb=np.zeros(12)
    Bb[10]=PCM_TO_RHO/TAU_V  # pcm-command input
    Cy=np.zeros(12); Cy[11]=1
    return Aa,Bb,Cy


def closed_loop_matrices(kp,ki):
    Ap,Bp,Cy=augmented_plant()
    # controller output (pcm) = Kp*error + Ki*z, zdot=error,
    # error=deltaTsp-deltaTmeas
    Acl=np.zeros((13,13))
    Acl[:12,:12]=Ap-np.outer(Bp,kp*Cy)
    Acl[:12,12]=Bp*ki
    Acl[12,:12]=-Cy
    Bsp=np.r_[Bp*kp, 1.0]
    return Acl,Bsp


def linear_response(kp,ki,t_post):
    Acl,Bsp=closed_loop_matrices(kp,ki)
    if np.max(np.linalg.eigvals(Acl).real)>=0: return None
    sol=solve_ivp(lambda t,x: Acl@x+Bsp*STEP_K,
                  (0,t_post[-1]),np.zeros(13),method='BDF',t_eval=t_post,
                  rtol=5e-8,atol=1e-10)
    return sol.y if sol.success else None


def objective(gains):
    kp,ki=gains
    Ac,_=closed_loop_matrices(kp,ki)
    if np.max(np.linalg.eigvals(Ac).real)>-0.0001: return 1e5
    ts=np.linspace(0,250,151)
    x=linear_response(kp,ki,ts)
    if x is None: return 1e5
    e=STEP_K-x[11]
    cmd=kp*e+ki*x[12]
    # emphasize tracking, penalize excessive reactivity command
    return np.trapezoid((e/STEP_K)**2,ts) + 0.3*np.trapezoid((cmd/1.0)**2,ts)


def margins(kp,ki):
    Ap,Bp,Cy=augmented_plant()
    w=np.logspace(-4,3,3000)
    H=np.array([Cy@np.linalg.solve(1j*wi*np.eye(12)-Ap,Bp) for wi in w]).ravel()
    L=(kp+ki/(1j*w))*H
    # from low to high frequency locate gain crossover
    mag=np.abs(L); phase=np.unwrap(np.angle(L))
    idx=np.where((mag[:-1]>=1)&(mag[1:]<1))[0]
    if not len(idx): return float('nan'),float('nan')
    j=idx[-1]
    f=(0-np.log(mag[j]))/(np.log(mag[j+1])-np.log(mag[j]))
    wc=w[j]*(w[j+1]/w[j])**f
    ph=phase[j]+f*(phase[j+1]-phase[j])
    pm=180+np.degrees(ph)
    return wc,pm


def simulate_nonlinear(kp,ki,t_post):
    # Full nonlinear reactor; x_abs=[10 core states, applied rho, actual sensor T, integrated error]
    x0=np.r_[m.Y0,0.0,m.TC2_REF,0.0]
    def rhs(t,x):
        e=(m.TC2_REF+STEP_K)-x[11]
        cmd_pcm=kp*e+ki*x[12]
        core_dot=m.ode(t,x[:10],x[10])
        drho=(PCM_TO_RHO*cmd_pcm-x[10])/TAU_V
        dmeas=(x[9]-x[11])/TAU_M
        return np.r_[core_dot,drho,dmeas,e]
    sol=solve_ivp(rhs,(0,t_post[-1]),x0,t_eval=t_post,
                  method='Radau',rtol=2e-8,atol=2e-10,max_step=2)
    if not sol.success: raise RuntimeError(sol.message)
    return sol.y


def main():
    # Initial full-model candidates: evaluate on all 13 states rather than a FOPDT.
    # Conservative selection prioritizes small, well-damped setpoint changes.
    # The controller is NOT globally optimal or validated for a real PWR.
    print("Full-model candidate screening (illustrative weighted cost):")
    for g in [(10.0,0.5),(20.0,1.0),(30.0,2.0)]:
        print(f"  Kp={g[0]:.1f}, Ki={g[1]:.1f}: cost={objective(g):.3f}, phase margin={margins(*g)[1]:.1f} deg")
    kp,ki=30.0,2.0
    Ac,_=closed_loop_matrices(kp,ki)
    poles=np.linalg.eigvals(Ac)
    wc,pm=margins(kp,ki)
    t_post=np.linspace(0,T_END-STEP_TIME,1300)
    x=linear_response(kp,ki,t_post)
    xn=simulate_nonlinear(kp,ki,t_post)
    out_lin=x[11]
    out_nl=xn[11]-m.TC2_REF
    maxdiff=np.max(np.abs(out_lin-out_nl))
    e=STEP_K-out_lin
    cmd_pcm=kp*e+ki*x[12]
    full_t=np.r_[np.linspace(0,STEP_TIME,100,endpoint=False),t_post+STEP_TIME]
    def prefix(arr): return np.r_[np.zeros(100),arr]
    print('FULL MODEL PI | LINEARIZED 10-STATE REACTOR + ACTUATOR + SENSOR')
    print(f'Kp={kp:.6f} pcm/K, Ki={ki:.6f} pcm/(K s), Ti={kp/ki:.3f} s')
    print(f'Worst closed-loop pole real part: {max(poles.real):.6g} s^-1')
    print(f'Loop gain crossover: {wc:.6g} rad/s; approximate phase margin: {pm:.2f} deg')
    print(f'Max nonlinear vs linear temp deviation for {STEP_K:g} K step: {maxdiff:.4g} K')
    print(f'Max rod command: {np.max(abs(cmd_pcm)):.5f} pcm')
    print('All gains are provisional; not a qualified PWR controller.')
    fig,ax=plt.subplots(3,1,figsize=(10,10),sharex=True,constrained_layout=True)
    ax[0].plot(full_t,np.r_[np.zeros(100),np.full(len(t_post),STEP_K)],'--',label='Setpoint change')
    ax[0].plot(full_t,prefix(out_lin),label='Full linear 13-state closed loop')
    ax[0].plot(full_t,prefix(out_nl),':',label='Nonlinear 13-state closed loop')
    ax[0].set_ylabel('Measured outlet ΔT [K]');ax[0].legend()
    ax[1].plot(full_t,prefix(cmd_pcm),label='Commanded rod reactivity')
    ax[1].plot(full_t,prefix(x[10]/PCM_TO_RHO),label='Applied reactivity')
    ax[1].set_ylabel('Rod reactivity [pcm]');ax[1].legend()
    ax[2].plot(full_t,prefix(out_nl-out_lin))
    ax[2].set_ylabel('Nonlinear − linear [K]');ax[2].set_xlabel('Time [s]')
    for a in ax: a.grid(alpha=.3);a.axvline(STEP_TIME,color='gray',ls=':')
    fig.suptitle('Model 1: PI tuned on full state-space plant (educational benchmark)')
    dst=Path(__file__).resolve().parent/'model_1_full_model_PI.png'
    fig.savefig(dst,dpi=150)
    print('Saved:',dst)

if __name__=='__main__': main()
