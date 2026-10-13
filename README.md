# Model 1 — Interactive Reduced-Order PWR Core Simulator

**Educational only.** This is an unvalidated research/portfolio benchmark, **not a nuclear-plant operating tool**.

## Run

From the project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will open a local browser tab. Modify the sidebar controls to explore a PI-controlled setpoint change, inlet-temperature/flow disturbance, or prescribed open-loop rod-reactivity change. Export the trajectory to CSV.

## Architecture

- `models/reactor_model_1.py` — ten core ODE states (normalized power, six precursors, fuel and two coolant nodes); provisional constants copied from earlier development model.
- `models/controller.py` — PI gains and first-order actuator/sensor assumptions.
- `models/disturbances.py` — step scenario inputs.
- `simulation/simulate.py` — reusable thirteen-state nonlinear solver and metrics.
- `app.py` — Streamlit / Plotly interface.
- `scripts/` — preserved previous exploratory and verification scripts.

The interface defaults to `Kp=30 pcm/K`, `Ki=2 pcm/(K·s)`, two-second actuator lag, one-second sensor lag, and a 100%-power calibrated starting state.

## Limitations

- The literature coefficients originate from Rahim et al. (2026), Table 2; **lambda_4** may contain a transcription or publication anomaly. The source's six-group values and temperature coefficients are not established as representative of AP1000.
- The coolant model is a simplified two-well-mixed-node conservation model rather than the exact paper thermal model.
- The current controller directly commands an idealized reactivity contribution, not a modeled rod position. No actuator saturation or anti-windup.
- The UI allows only deliberately small educational perturbations; these ranges are not safety limits.
- No pressure/SG/pressurizer dynamics, protection logic, or operator guidance.

## Reference

Rahim et al. (2026), *Modeling and control of a nuclear pressurized water reactor using advanced model predictive control strategies*, Next Energy 11, 100597. DOI: 10.1016/j.nxener.2026.100597.
