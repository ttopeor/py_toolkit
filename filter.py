#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light-weight Kalman smoother for fixed-size vectors.

Typical usage
-------------
cfg = yaml.safe_load(open("config.yaml"))
flt = DataFilter(cfg, cov_name="measurement_cov", smooth_factor=0.05)

flt.set_offsets([0.1, 0, 0, 0, 0, 0])
smoothed = flt.filter_data(raw_vector)
"""

from __future__ import annotations
from typing import Sequence, Mapping
import numpy as np


class DataFilter:
    """
    Simple Kalman filter wrapper for smoothing fixed-length vectors.

    Parameters
    ----------
    config : Mapping[str, Sequence | list | np.ndarray]
        Configuration dict / YAML that may contain a covariance entry.
    smooth_factor : float, optional
        Scales the *process* noise covariance; smaller → heavier smoothing.
    cov_name : str | None, optional
        Key in ``config`` that stores the measurement covariance (list/array).
        If ``None`` or missing, an identity matrix * default_cov_val is used.
    default_cov_val : float, optional
        Variance value for default measurement covariance (diagonal).

    Notes
    -----
    * State and measurement dimensions are assumed identical (= ``state_dim``).
    * This is a basic linear Kalman filter with identity state transition
      and no control input.
    """

    def __init__(
        self,
        config: Mapping[str, Sequence | np.ndarray],
        *,
        state_dim: int = 6,
        smooth_factor: float = 0.1,
        cov_name: str | None = None,
        default_cov_val: float = 1e-2,
    ) -> None:
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        self.state_dim = state_dim
        self.offsets: np.ndarray = np.zeros(state_dim, dtype=np.float32)
        self.prev_estimate: np.ndarray = np.zeros(state_dim, dtype=np.float32)
        self.initialized: bool = False

        # ── measurement covariance ───────────────────────────────────
        cov = None
        if cov_name is not None:
            cov = config.get(cov_name)

        if cov is None:  # fallback
            cov = np.eye(state_dim, dtype=np.float32) * default_cov_val

        self.measurement_covariance: np.ndarray = np.asarray(cov, dtype=np.float32)
        if self.measurement_covariance.shape != (state_dim, state_dim):
            raise ValueError(
                f"Measurement covariance must be ({state_dim}, {state_dim})"
            )

        # ── Kalman parameters ─────────────────────────────────────────
        self.estimate_covariance: np.ndarray = self.measurement_covariance.copy()
        self.process_covariance: np.ndarray = self.measurement_covariance * smooth_factor

        # Pre-allocate identity for speed
        self._I: np.ndarray = np.eye(state_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_offsets(self, offset_array: Sequence[float]) -> None:
        """Set static offsets to be subtracted from incoming measurements."""
        offset_arr = np.asarray(offset_array, dtype=np.float32)
        if offset_arr.shape != (self.state_dim,):
            raise ValueError(f"offset_array must have shape ({self.state_dim},)")
        self.offsets = offset_arr

    def filter_data(self, input_array: Sequence[float]) -> np.ndarray:
        """
        Apply offsets and Kalman smoothing to an input vector.

        Parameters
        ----------
        input_array : Sequence[float]
            Raw measurement with length = ``state_dim``.

        Returns
        -------
        np.ndarray
            Smoothed vector of shape (state_dim,).
        """
        z = np.asarray(input_array, dtype=np.float32)
        if z.shape != (self.state_dim,):
            raise ValueError(f"input_array must have shape ({self.state_dim},)")

        # Apply offsets
        z_adj = z - self.offsets

        if not self.initialized:
            self.prev_estimate = z_adj.copy()
            self.initialized = True
            return z_adj

        return self._kalman_step(z_adj)

    # ------------------------------------------------------------------
    # Internal: one Kalman iteration
    # ------------------------------------------------------------------
    def _kalman_step(self, measurement: np.ndarray) -> np.ndarray:
        # Prediction (identity model → state unchanged)
        pred_est = self.prev_estimate
        pred_cov = self.estimate_covariance + self.process_covariance

        # Innovation
        innovation = measurement - pred_est
        innov_cov = pred_cov + self.measurement_covariance

        # Kalman gain
        K = pred_cov @ np.linalg.inv(innov_cov)

        # Update
        current_est = pred_est + K @ innovation
        self.estimate_covariance = (self._I - K) @ pred_cov
        self.prev_estimate = current_est

        return current_est
