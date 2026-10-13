"""Model 1: analytic Jacobian and small-signal validation.

Place next to model_1_open_loop.py. Both are educational benchmark models,
not validated AP1000 plant/control models.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal
import model_1_open_loop as m


def build_matrices():
    """State order: [Pn, C1..C6, Tf, Tc1, Tc2]. Input: delta rho_rod."""
    A = np.zeros((10, 10), dtype=float)
    B = np.zeros((10, 1), dtype=float)
    C = np.zeros((1, 10), dtype=float)
    D = np.zeros((1, 1), dtype=float)

    # Normalized six-group kinetics at Pn0=1, rho0=0.
    A[0, 0] = -m.BETA / m.LAMBDA
    A[0, 1:7] = m.BETA_I / m.LAMBDA
    A[0, 7] = m.ALPHA_F / m.LAMBDA
    A[0, 8:10] = m.ALPHA_C / m.LAMBDA
    B[0, 0] = 1.0 / m.LAMBDA

    for i in range(6):
        A[i+1, 0] = m.DECAY[i]
        A[i+1, i+1] = -m.DECAY[i]

    hf = m.M_F * m.CP_F
    hc1 = m.M_C1 * m.CP_C
    hc2 = m.M_C2 * m.CP_C

    # Fuel node.
    A[7, 0] = m.Q_RATED / hf
    A[7, 7] = -2.0 * m.U / hf
    A[7, 8] = m.U / hf
    A[7, 9] = m.U / hf

    # Lower coolant node.
    A[8, 7] = m.U / hc1
    A[8, 8] = -(m.U + m.F) / hc1

    # Upper coolant / outlet node.
    A[9, 7] = m.U / hc2
    A[9, 8] = m.F / hc2
    A[9, 9] = -(m.U + m.F) / hc2

    C[0, 9] = 1.0
    return A, B, C, D


def finite_difference_jacobian(x0, u0=0.0):
    """Independent numerical Jacobian check of the existing ODE."""
    n = len(x0)
    A_num = np.empty((n, n))
    # Different numerical perturbation sizes for p.u. and Kelvin states.
    steps = np.array([1e-5]*7 + [1e-3]*3)
    for j in range(n):
        xp = x0.copy(); xm = x0.copy()
        xp[j] += steps[j]; xm[j] -= steps[j]
        A_num[:,j] = (m.ode(0,xp,u0)-m.ode(0,xm,u0))/(2*steps[j])
    eps_u = 1e-7
    B_num = ((m.ode(0,x0,u0+eps_u)-m.ode(0,x0,u0-eps_u))/(2*eps_u)).reshape(-1,1)
    return A_num, B_num


def simulate_step(A, B, t_end=200.0, t_step=20.0, pcm=1.0):
    du = pcm * 1e-5
    t_pre = np.linspace(0, t_step, 201)
    t_post = np.linspace(t_step, t_end, 1801)
    dx0 = np.zeros(10)

    def lin_rhs(t, dx):
        return A @ dx + B[:, 0] * du

    lin = solve_ivp(lin_rhs, (t_step,t_end), dx0, method='Radau',
                    t_eval=t_post, rtol=1e-9, atol=1e-11, max_step=1.0)
    nonlin = solve_ivp(lambda t, y: m.ode(t,y,du), (t_step,t_end), m.Y0,
                       method='Radau', t_eval=t_post, rtol=1e-9,
                       atol=1e-11, max_step=1.0)
    if not lin.success or not nonlin.success:
        raise RuntimeError(f'Integration failed: {lin.message}; {nonlin.message}')

    t = np.r_[t_pre[:-1], t_post]
    y_lin = np.column_stack([np.zeros((10,len(t_pre)-1)), lin.y])
    y_non = np.column_stack([np.tile(m.Y0[:,None],(1,len(t_pre)-1)), nonlin.y])
    return t, y_lin, y_non


def main():
    A, B, C, D = build_matrices()
    An, Bn = finite_difference_jacobian(m.Y0)
    print('MODEL 1 LINEARIZATION')
    print(f'Max |A_analytic - A_finite_difference| = {np.max(np.abs(A-An)):.3e} (mixed units)')
    print(f'Max |B_analytic - B_finite_difference| = {np.max(np.abs(B-Bn)):.3e} (mixed units)')
    print('Open-loop eigenvalues (1/s):')
    print(np.sort_complex(np.linalg.eigvals(A)))
    print('Output: delta Tout = delta Tc2; input: delta rod reactivity (dimensionless).')
    # Prefer state-space for numerical simulation: forming a high-order
    # numerator/denominator polynomial can be poorly conditioned.
    Gp_ss = signal.StateSpace(A, B, C, D)
    print('Gp(s) = C(sI-A)^(-1)B, stored numerically as a 10-state system.')

    t, xlin, xnon = simulate_step(A, B, t_end=m.T_END,
                                  t_step=m.T_STEP, pcm=m.ROD_STEP_PCM)
    lin_tout = (C @ xlin).ravel()
    non_tout = xnon[9] - m.TC2_REF
    lin_power = xlin[0]
    non_power = xnon[0] - 1.0

    after = t >= m.T_STEP
    err = non_tout[after] - lin_tout[after]
    print(f'Outlet temp maximum absolute difference = {np.max(np.abs(err)):.6e} K')
    print(f'Outlet temp RMSE = {np.sqrt(np.mean(err**2)):.6e} K')
    print(f'Final outlet delta: nonlinear {non_tout[-1]:.8f} K, linear {lin_tout[-1]:.8f} K')
    print(f'Final normalized-power delta: nonlinear {non_power[-1]:.8f}, linear {lin_power[-1]:.8f}')

    fig, axs = plt.subplots(3,1,figsize=(9,10),sharex=True,constrained_layout=True)
    axs[0].plot(t, non_tout, label='Nonlinear reactor ODE', lw=2)
    axs[0].plot(t, lin_tout, '--', label='Linearized state-space Gp', lw=1.6)
    axs[0].set_ylabel('Outlet deviation [K]')
    axs[0].legend()
    axs[1].plot(t, non_power, label='Nonlinear power deviation')
    axs[1].plot(t, lin_power, '--',label='Linearized power deviation')
    axs[1].set_ylabel('Power deviation [p.u.]')
    axs[1].legend()
    axs[2].plot(t, non_tout-lin_tout)
    axs[2].set_ylabel('Outlet model difference [K]')
    axs[2].set_xlabel('Time [s]')
    for ax in axs:
        ax.axvline(m.T_STEP, color='gray', ls=':')
        ax.grid(alpha=.3)
    fig.suptitle(f'Model 1 linearization validation: {m.ROD_STEP_PCM:+g} pcm')
    fig.savefig('model_1_linear_validation.png', dpi=160)
    plt.show()


if __name__=='__main__':
    main()
