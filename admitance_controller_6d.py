import numpy as np
from typing import Sequence
from py_toolkit.math_tools import parent_to_child

class AdmittanceController6D:
    def __init__(
        self,
        M: Sequence[float],
        B: Sequence[float],
        init_p: Sequence[float],
    ):
        assert len(M) == len(B) == 6, "Lengths of M and B must be 6"
        # Convert to NumPy arrays for vectorised maths
        self.M = np.array(M, dtype=float)
        self.B = np.array(B, dtype=float)
        self.p = np.copy(np.array(init_p, dtype=float))

        self.F_ext_d = np.zeros(6, dtype=float)
        self._last_v = np.zeros(6, dtype=float)


    def set_desired_force(self, F_ext_d: Sequence[float]) -> None:
        assert len(F_ext_d) == 6, "Desired force must have length 6"
        self.F_ext_d = np.array(F_ext_d, dtype=float)

    def update(self, F_ext: Sequence[float], dt: float) -> np.ndarray:
        F_ext = np.array(F_ext, dtype=float)
        e_f = self.F_ext_d - F_ext

        # M * a + B * v = e_f  ->  a = (e_f - B * v) / M
        a = (e_f - self.B * self._last_v) / self.M
        v = self._last_v + a * dt
        delta_p = v * dt
        
        self.p = parent_to_child(self.p, delta_p)

        self._last_v = v
        return self.p
