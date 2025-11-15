import numpy as np
from typing import Sequence
from py_toolkit.math_tools import calculate_delta_position


class ImpedanceController6D:
    def __init__(self, B: Sequence[float], K: Sequence[float], init_p: Sequence[float]):
        assert len(B) == len(K) == 6, "B, K must have length 6"
        self.B = np.array(B, dtype=float)
        self.K = np.array(K, dtype=float)

        self.p_d = np.copy(np.array(init_p, dtype=float))
        self._last_p = np.copy(np.array(init_p, dtype=float))

    def set_desired_position(self, p_d: Sequence[float]) -> None:
        assert len(p_d) == 6, "Desired position must have length 6"
        self.p_d = np.array(p_d, dtype=float)

    def update(self, p: Sequence[float], dt: float) -> np.ndarray:
        p = np.array(p, dtype=float)

        e_p = calculate_delta_position(p, self.p_d)
        e_v = calculate_delta_position(p, self._last_p) / dt

        F_cmd = self.K * e_p + self.B * e_v
        self._last_p = p

        return F_cmd
