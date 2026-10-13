"""Educational reduced-order PWR core: open-loop neutronic/thermal coupling.

Not an AP1000-certified model or a reactor operating/control procedure.
Six-group constants and feedback coefficients are transcribed from Rahim et al.
(2026), Next Energy, Table 2, DOI 10.1016/j.nxener.2026.100597.
NOTE: Table 2 prints lambda_4=0.03013 s^-1 and repeats the beta_4 label.
These may be typographical errors; both require verification against the
underlying Vajpayee reference before any quantitative validation.

Thermal equations are the project's own two well-mixed coolant-node balances,
NOT the thermal equations of Rahim et al. No pressure or rod-drive dynamics.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# A. THERMAL MODEL: Model 0 nominal conditions, Model 1 topology
# ============================================================
Q_RATED = 3.40e9             # W, modeling reference
T_IN = 553.85               # K
M_DOT = 14275.6             # kg/s
CP_C = 5590.0               # J/(kg K)
M_C = 11665.8               # kg total active-core coolant inventory
M_F = 96241.0               # kg fuel
CP_F = 308.0                # J/(kg K)
TF_REF = 900.0              # K, assumed effective reference fuel temperature
M_C1 = M_C2 = M_C / 2.0    # kg, equal well-mixed inventories
F = M_DOT * CP_C            # W/K, flow heat-capacity rate

# At 100% power, core-outlet temperature follows conservation of energy.
TOUT_REF = T_IN + Q_RATED / F
# Calibrate UA_1 = UA_2 = U to match TF_REF and TOUT_REF at full power.
r = np.sqrt((TF_REF - T_IN) / (TF_REF - TOUT_REF)) - 1.0
U = F * r                    # W/K for EACH node
TC1_REF = (T_IN + r * TF_REF) / (1.0 + r)
TC2_REF = TOUT_REF

# ============================================================
# B. NEUTRON KINETICS: published benchmark constants, provisional
# ============================================================
LAMBDA = 3.0e-5  # s, prompt neutron generation time (NOT lambda decay)
DECAY = np.array([0.012437, 0.03050, 0.11141, 0.03013, 1.12866, 3.0130])  # s^-1
BETA_I = np.array([2.15e-4, 1.424e-3, 1.274e-3, 2.568e-3, 7.48e-4, 2.73e-4])
BETA = float(BETA_I.sum())

ALPHA_F = -2.16e-5    # (delta k/k)/K; illustrative paper value
ALPHA_C = -1.80e-4    # (delta k/k)/K, applied to EACH coolant node per paper

# ============================================================
# C. EXPERIMENT: no PI controller, no rod-drive model
# ============================================================
T_STEP = 20.0              # s
T_END = 200.0              # s
ROD_STEP_PCM = 1.0         # pcm; try 0 or small +/- values for sensitivity
RHO_STEP = ROD_STEP_PCM * 1.0e-5  # 1 pcm = 10^-5 delta k/k


def total_reactivity(tf, tc1, tc2, rho_rod):
    """Dimensionless reactivity relative to the calibrated rated equilibrium."""
    return (rho_rod
            + ALPHA_F * (tf - TF_REF)
            + ALPHA_C * ((tc1 - TC1_REF) + (tc2 - TC2_REF)))


def ode(t, y, rho_rod):
    """State order: Pn, C1 ... C6, Tf, Tc1, Tc2."""
    pn = y[0]
    precursors = y[1:7]
    tf, tc1, tc2 = y[7:10]
    rho = total_reactivity(tf, tc1, tc2, rho_rod)

    # Six-group normalized point kinetics
    dpn_dt = ((rho - BETA) * pn + np.dot(BETA_I, precursors)) / LAMBDA
    dc_dt = DECAY * (pn - precursors)

    # First-principles energy balances with two *well-mixed* nodes in series
    q1 = U * (tf - tc1)
    q2 = U * (tf - tc2)
    dtf_dt = (Q_RATED * pn - q1 - q2) / (M_F * CP_F)
    dtc1_dt = (q1 + F * (T_IN - tc1)) / (M_C1 * CP_C)
    dtc2_dt = (q2 + F * (tc1 - tc2)) / (M_C2 * CP_C)
    return np.concatenate(([dpn_dt], dc_dt, [dtf_dt, dtc1_dt, dtc2_dt]))


Y0 = np.array([1.0, *([1.0] * 6), TF_REF, TC1_REF, TC2_REF], dtype=float)


def run_segment(t0, t1, y0, rho_rod, npts):
    """Radau handles the stiff prompt/delayed time scales."""
    t_eval = np.linspace(t0, t1, npts)
    result = solve_ivp(
        lambda time, state: ode(time, state, rho_rod),
        (t0, t1), y0, method="Radau", t_eval=t_eval,
        rtol=1e-8, atol=1e-10, max_step=1.0
    )
    if not result.success:
        raise RuntimeError(f"ODE integration failed: {result.message}")
    return result


def main():
    # Verify equilibrium of all 10 states before applying any transient.
    residual = ode(0.0, Y0, 0.0)
    assert np.all(np.abs(residual) < 1e-8), f"Invalid equilibrium: {residual}"
    assert np.isclose(TC2_REF, TOUT_REF)
    assert np.isclose(Q_RATED, F * (TC2_REF - T_IN), rtol=1e-10)
    assert 1.0 - BETA > 0.0

    # Integrate separately either side of the discontinuity; don't make the
    # solver guess where the external rod reactivity jumps.
    pre = run_segment(0.0, T_STEP, Y0, 0.0, 200)
    post = run_segment(T_STEP, T_END, pre.y[:, -1], RHO_STEP, 1801)
    t = np.r_[pre.t, post.t[1:]]
    y = np.column_stack((pre.y, post.y[:, 1:]))

    pn = y[0]
    tf, tc1, tc2 = y[7], y[8], y[9]
    precursors = y[1:7]
    rho_rod = np.where(t < T_STEP, 0.0, RHO_STEP)
    rho_temp = ALPHA_F * (tf - TF_REF) + ALPHA_C * ((tc1-TC1_REF) + (tc2-TC2_REF))
    rho_total = rho_rod + rho_temp

    print("MODEL 1: OPEN-LOOP SIX-GROUP REACTOR")
    print("Source kinetics: Rahim et al. (2026), Table 2 (provisional)")
    print(f"beta_total         = {BETA:.8f}")
    print(f"Lambda             = {LAMBDA:.2e} s")
    print(f"UA_1 = UA_2        = {U/1e6:.6f} MW/K")
    print(f"Tc1,ref            = {TC1_REF:.6f} K")
    print(f"Tc2,ref = Tout,ref = {TC2_REF:.6f} K")
    print(f"Initial derivative max absolute = {np.max(abs(residual)):.3e} (mixed units)")
    print(f"Reactivity step    = {ROD_STEP_PCM:+.3f} pcm at t={T_STEP:g} s")
    print(f"Final normalized power      = {pn[-1]:.6f} p.u.")
    print(f"Final fuel temperature      = {tf[-1]:.6f} K")
    print(f"Final outlet temperature    = {tc2[-1]:.6f} K")
    print(f"Final total reactivity      = {rho_total[-1]/1e-5:.6f} pcm")
    print("Caution: equilibrium by 200 s is not guaranteed; precursors have long time constants.")

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axs[0].plot(t, pn, label="Normalized reactor power")
    axs[0].axhline(1.0, color="gray", linestyle="--", lw=1, label="Rated reference")
    axs[0].set_ylabel("Power [p.u.]")
    axs[0].legend()

    axs[1].plot(t, rho_rod / 1e-5, linestyle="--", label="Inserted rod reactivity")
    axs[1].plot(t, rho_temp / 1e-5, label="Temperature feedback")
    axs[1].plot(t, rho_total / 1e-5, label="Total reactivity")
    axs[1].set_ylabel("Reactivity [pcm]")
    axs[1].legend()

    axs[2].plot(t, tf - TF_REF, label="Fuel temperature deviation")
    axs[2].plot(t, tc1 - TC1_REF, label="Lower coolant deviation")
    axs[2].plot(t, tc2 - TC2_REF, label="Outlet temperature deviation")
    axs[2].set_ylabel("Temperature change [K]")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()
    for ax in axs:
        ax.axvline(T_STEP, color="gray", linestyle=":", lw=1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Model 1 | Open-loop rod reactivity step (educational benchmark)")

    out_path = Path(__file__).resolve().with_name("model_1_open_loop.png")
    fig.savefig(out_path, dpi=170)
    print(f"Saved figure to {out_path}")
    plt.show()

    # Optional separate precursor plot, useful during initial verification.
    plt.figure(figsize=(9, 5))
    for i in range(6):
        plt.plot(t, precursors[i], label=f"C{i+1}")
    plt.axvline(T_STEP, color="gray", linestyle=":")
    plt.title("Delayed-neutron precursor groups (normalized)")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized precursor concentration")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
