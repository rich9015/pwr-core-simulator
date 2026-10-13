"""Model 1: provisional actuator + reactor + sensor FOPDT evaluation.

Put next to model_1_open_loop.py and model_1_linear_validation.py.
All parameters are educational modeling assumptions, not qualified PWR data.

Plant-side path: commanded rod reactivity [pcm] -> Gv -> nonlinear/linear
reactor -> Gm -> measured core outlet temperature [K].
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import model_1_open_loop as m
from model_1_linear_validation import build_matrices

TAU_V = 2.0  # s, illustrative *effective* command-to-reactivity lag
TAU_M = 1.0  # s, illustrative temperature measurement lag
STEP_PCM = 1.0
STEP_TIME = 20.0
END_TIME = 800.0
PCM_TO_RHO = 1e-5

# Actuator and output augment original 10-state nonlinear core:
# state=[Pn,C1..C6,Tf,Tc1,Tc2,rho_applied,Tmeas]
Y0_AUG = np.r_[m.Y0, 0.0, m.TC2_REF]


def commanded_pcm(t):
    return 0.0 if t < STEP_TIME else STEP_PCM


def rhs_aug(t, y, rho_command):
    dy_reactor = m.ode(t, y[:10], y[10])
    d_rho = (rho_command - y[10]) / TAU_V
    d_tmeas = (y[9] - y[11]) / TAU_M
    return np.r_[dy_reactor, d_rho, d_tmeas]


def integrate_piecewise(rhs, initial, dt=0.5):
    """Apply exact zero pre-step equilibrium and integrate post-step with Radau."""
    tpre = np.arange(0.0, STEP_TIME, dt)
    tpost = np.arange(STEP_TIME, END_TIME + 0.5*dt, dt)
    sol = solve_ivp(rhs, (STEP_TIME, END_TIME), initial, method='Radau',
                    t_eval=tpost, rtol=1e-9, atol=1e-11, max_step=2.0)
    if not sol.success:
        raise RuntimeError(sol.message)
    tt = np.r_[tpre, sol.t]
    xx = np.column_stack((np.tile(initial[:, None], (1, len(tpre))), sol.y))
    return tt, xx


def fopdt_step(t, gain, tau, theta):
    """Response in K to STEP_PCM input, where gain is K/pcm."""
    active_time = np.maximum(np.asarray(t) - STEP_TIME - theta, 0.0)
    return gain * STEP_PCM * (-np.expm1(-active_time / tau))


def main():
    # Independently built exact linearized augmented state-space, with input in pcm.
    A, B, C, _ = build_matrices()
    Aa = np.zeros((12, 12))
    Aa[:10, :10] = A
    Aa[:10, 10] = B[:, 0]   # reactor sees applied reactivity
    Aa[10, 10] = -1.0 / TAU_V
    Aa[11, 9] = 1.0 / TAU_M
    Aa[11, 11] = -1.0 / TAU_M
    Ba = np.zeros(12)
    Ba[10] = PCM_TO_RHO / TAU_V  # commanded input in pcm
    Ca = np.zeros(12)
    Ca[11] = 1.0
    gain_steady = -Ca @ np.linalg.solve(Aa, Ba)  # K/pcm

    # Simulate exact linear 12-state model in deviation variables.
    t, lin = integrate_piecewise(lambda tt, yy: Aa @ yy + Ba * STEP_PCM,
                                  np.zeros(12))
    y_linear = lin[11]

    # Validate the same step against augmented nonlinear reactor equations.
    tn, nonlinear = integrate_piecewise(
        lambda tt, yy: rhs_aug(tt, yy, STEP_PCM*PCM_TO_RHO), Y0_AUG)
    y_nonlinear = nonlinear[11] - m.TC2_REF
    assert np.array_equal(t, tn)

    # Fit only post-step samples, scale residuals for numerical robustness.
    active = t >= STEP_TIME
    def res(p):
        return (fopdt_step(t[active], *p) - y_linear[active])
    fit = least_squares(res, x0=[float(gain_steady), 8.0, 1.0],
                        bounds=([0., .001, 0.], [10., 2000., 100.]),
                        x_scale=[0.02, 10., 1.0],
                        xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=5000)
    K, tau, theta = fit.x
    y_fopdt = fopdt_step(t, K, tau, theta)
    err = y_linear[active] - y_fopdt[active]
    err_nl = y_nonlinear[active] - y_linear[active]

    # Sensitivity of fit to the transient fitting horizon (200 s only).
    short_mask = (t >= STEP_TIME) & (t <= 200.0)
    fit_short = least_squares(
        lambda p: fopdt_step(t[short_mask], *p) - y_linear[short_mask],
        x0=fit.x, bounds=([0., .001, 0.], [10., 2000., 100.]),
        x_scale=[0.02,10.,1.0],max_nfev=5000)

    print('MODEL 1: Gv - Gp - Gm STEP IDENTIFICATION')
    print('Actuator input: commanded rod reactivity [pcm]')
    print('Sensor output: measured outlet temperature deviation [K]')
    print(f'tau_v = {TAU_V:.3f} s (illustrative; not validated rod-drive data)')
    print(f'tau_m = {TAU_M:.3f} s (illustrative sensor lag)')
    print(f'Step = +{STEP_PCM:g} pcm at {STEP_TIME:g} s; horizon = {END_TIME:g} s')
    print(f'Exact linearized DC gain = {gain_steady:.9f} K/pcm')
    print(f'FOPDT fit: K={K:.9f} K/pcm, tau={tau:.6f} s, theta={theta:.6f} s')
    print(f'FOPDT RMSE: {np.sqrt(np.mean(err**2)):.8f} K')
    print(f'FOPDT max absolute error: {np.max(np.abs(err)):.8f} K')
    print(f'Normalized FOPDT RMSE: {np.sqrt(np.mean(err**2))/abs(gain_steady*STEP_PCM)*100:.3f}% of steady step')
    print(f'Nonlinear vs linear RMSE: {np.sqrt(np.mean(err_nl**2)):.8f} K')
    print(f'Fit using only 20-200 s: K={fit_short.x[0]:.9f}, tau={fit_short.x[1]:.5f}, theta={fit_short.x[2]:.5f}')
    print(f'FOPDT/steady gain ratio: {K/gain_steady:.5f}')
    print(f'FOPDT fit success: {fit.success}')

    out = Path(__file__).resolve().parent
    fig, axs = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
    axs[0].plot(t, y_linear, lw=2, label='Linearized reactor + actuator + sensor')
    axs[0].plot(t, y_nonlinear, '--', lw=1.4, label='Nonlinear reactor + actuator + sensor')
    axs[0].plot(t, y_fopdt, ':', lw=2.2, label='FOPDT fit')
    axs[0].axhline(gain_steady*STEP_PCM, color='gray', ls='--', alpha=.55, label='Exact DC prediction')
    axs[0].set(xlabel='Time [s]', ylabel='Measured outlet ΔT [K]', title='Full-horizon response')
    axs[0].legend(fontsize=8)
    axs[1].plot(t, y_linear, lw=2, label='Linear model')
    axs[1].plot(t, y_fopdt, '--', label='FOPDT')
    axs[1].set(xlim=(15,150), xlabel='Time [s]', ylabel='Measured outlet ΔT [K]', title='Transient detail')
    axs[1].legend()
    axs[2].plot(t, y_linear-y_fopdt, label='Linear model minus FOPDT')
    axs[2].set(xlabel='Time [s]', ylabel='Fit residual [K]', title='Approximation error')
    for a in axs:
        a.axvline(STEP_TIME, color='gray', ls=':', lw=1)
        a.grid(alpha=.3)
    fig.suptitle('Model 1 | Evaluate whether FOPDT is suitable (provisional benchmark)')
    fig.savefig(out/'model_1_fopdt_validation.png',dpi=170)
    np.savetxt(out/'model_1_fopdt_step_data.csv',
               np.column_stack([t, np.where(t<STEP_TIME,0,STEP_PCM),
                                lin[10]/PCM_TO_RHO,y_linear,y_nonlinear,y_fopdt]),
               delimiter=',',comments='',header='Time_s,Reactivity_command_pcm,Reactivity_applied_pcm,Tmeas_linear_delta_K,Tmeas_nonlinear_delta_K,Tmeas_FOPDT_delta_K')
    plt.close(fig)

if __name__ == '__main__':
    main()
