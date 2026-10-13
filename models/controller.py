"""Provisional PI and first-order actuator/sensor model for educational tests."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ControlParameters:
    kp: float = 30.0  # pcm/K
    ki: float = 2.0   # pcm/(K s)
    tau_v: float = 2.0
    tau_m: float = 1.0
