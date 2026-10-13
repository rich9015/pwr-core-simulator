"""Inputs are piecewise-constant deviations from calibrated reference."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Scenario:
    duration: float = 400.0
    step_time: float = 50.0
    setpoint_step_k: float = 0.01
    inlet_step_k: float = 0.0
    flow_step_percent: float = 0.0
    rod_step_pcm: float = 0.0
    closed_loop: bool = True

    def active(self, t): return t >= self.step_time
    def values(self, t):
        active=self.active(t)
        return (self.setpoint_step_k if active else 0.0,
                self.inlet_step_k if active else 0.0,
                1.0+(self.flow_step_percent/100.0 if active else 0.0),
                self.rod_step_pcm if active else 0.0)
