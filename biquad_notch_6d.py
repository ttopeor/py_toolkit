#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import Sequence, Union, Optional

ArrayLike = Union[Sequence[float], np.ndarray]


class BiquadNotch6D:
    """
    6D Biquad Notch (RBJ cookbook) applied per axis independently.

    For each axis:
      y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

    Notch params:
      fs: sampling rate (Hz)
      f0: notch center (Hz) scalar or length-6
      Q : quality factor scalar or length-6 (smaller => wider notch)
      axis_mask: optional length-6 bool/0-1 enable mask
    """

    def __init__(
        self,
        fs: float,
        f0: ArrayLike,
        Q: ArrayLike,
        axis_mask: Optional[ArrayLike] = None,
    ):
        self.fs = float(fs)
        if self.fs <= 0.0:
            raise ValueError("fs must be > 0")

        self.f0 = self._to6(f0, "f0")
        self.Q = self._to6(Q, "Q")

        if axis_mask is None:
            self.axis_mask = np.ones(6, dtype=bool)
        else:
            m = np.asarray(axis_mask).reshape(-1)
            if m.size != 6:
                raise ValueError("axis_mask must be length-6")
            self.axis_mask = (m.astype(float) != 0.0)

        # coeffs
        self.b0 = np.ones(6, dtype=float)
        self.b1 = np.zeros(6, dtype=float)
        self.b2 = np.zeros(6, dtype=float)
        self.a1 = np.zeros(6, dtype=float)
        self.a2 = np.zeros(6, dtype=float)

        # states
        self.x1 = np.zeros(6, dtype=float)
        self.x2 = np.zeros(6, dtype=float)
        self.y1 = np.zeros(6, dtype=float)
        self.y2 = np.zeros(6, dtype=float)

        self._recompute()

    @staticmethod
    def _to6(x: ArrayLike, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.repeat(arr.item(), 6)
        if arr.size != 6:
            raise ValueError(
                f"{name} must be scalar or length-6, got {arr.size}")
        return arr.copy()

    def set_params(self, f0: Optional[ArrayLike] = None, Q: Optional[ArrayLike] = None) -> None:
        if f0 is not None:
            self.f0 = self._to6(f0, "f0")
        if Q is not None:
            self.Q = self._to6(Q, "Q")
        self._recompute()

    def _recompute(self) -> None:
        for i in range(6):
            f0 = float(self.f0[i])
            Q = float(self.Q[i])

            # bypass invalid params on that axis
            if f0 <= 0.0 or f0 >= 0.5 * self.fs or Q <= 0.0:
                self.b0[i], self.b1[i], self.b2[i] = 1.0, 0.0, 0.0
                self.a1[i], self.a2[i] = 0.0, 0.0
                continue

            w0 = 2.0 * math.pi * (f0 / self.fs)
            cosw0 = math.cos(w0)
            alpha = math.sin(w0) / (2.0 * Q)

            # RBJ Notch
            b0, b1, b2 = 1.0, -2.0 * cosw0, 1.0
            a0, a1, a2 = 1.0 + alpha, -2.0 * cosw0, 1.0 - alpha

            self.b0[i] = b0 / a0
            self.b1[i] = b1 / a0
            self.b2[i] = b2 / a0
            self.a1[i] = a1 / a0
            self.a2[i] = a2 / a0

    def reset(self, x0: Optional[ArrayLike] = None) -> None:
        if x0 is None:
            v = np.zeros(6, dtype=float)
        else:
            v = np.asarray(x0, dtype=float).reshape(-1)
            if v.size != 6:
                raise ValueError("x0 must be length-6")
        self.x1[:] = v
        self.x2[:] = v
        self.y1[:] = v
        self.y2[:] = v

    def update(self, x: ArrayLike) -> np.ndarray:
        xv = np.asarray(x, dtype=float).reshape(-1)
        if xv.size != 6:
            raise ValueError("input x must be length-6")

        y = np.empty(6, dtype=float)
        for i in range(6):
            if not self.axis_mask[i]:
                y[i] = xv[i]
                continue
            y[i] = (self.b0[i] * xv[i] +
                    self.b1[i] * self.x1[i] +
                    self.b2[i] * self.x2[i] -
                    self.a1[i] * self.y1[i] -
                    self.a2[i] * self.y2[i])

        # shift states
        self.x2[:] = self.x1
        self.x1[:] = xv
        self.y2[:] = self.y1
        self.y1[:] = y

        return y
