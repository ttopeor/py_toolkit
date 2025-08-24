#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light-weight Kalman smoother for fixed-size vectors with **arbitrary DoF**.

What changed vs. your original:
- **Arbitrary dimension support**: `state_dim` is optional. If omitted, it is
  inferred from the measurement covariance in `config` (if present) or lazily
  from the first input vector passed to `filter_data`. Offsets can also fix the
  dimension if set first.
- **Flexible covariance input**: Measurement covariance may be provided as:
  * an (N, N) full matrix,
  * a length-N 1D array/list interpreted as a diagonal,
  * a scalar (applied to the diagonal),
  * or omitted (falls back to identity * `default_cov_val`).
- **Numerical robustness**: Uses `np.linalg.solve` (no explicit matrix inverse);
  falls back to `pinv` if the innovation covariance is singular.
- **Reset options**: `reset(hard=False)` keeps the dimension; `reset(hard=True)`
  also forgets the dimension so the next call can re-infer it.

Typical usage
-------------
>>> import yaml
>>> cfg = yaml.safe_load(open("config.yaml"))
>>> flt = DataFilter(cfg, cov_name="measurement_cov", smooth_factor=0.05)  # no state_dim → infer
>>> flt.set_offsets([0.1, 0, 0, 0, 0, 0])  # also fixes DoF to 6 if not fixed yet
>>> smoothed = flt.filter_data(raw_vector)

You can also pin the dimension explicitly:
>>> flt = DataFilter(cfg, state_dim=8, cov_name="measurement_cov", smooth_factor=0.02)
"""

from __future__ import annotations
from typing import Sequence, Mapping, Any, Optional, Union
import numpy as np


Number = Union[int, float, np.number]


class DataFilter:
    """
    Simple linear Kalman filter wrapper for smoothing fixed-length vectors.

    Assumptions
    -----------
    * Linear, time-invariant model with identity state transition (no control).
    * Measurement matrix is identity (state and measurement spaces coincide).
    * State dimensionality N is either specified or inferred (see below).

    Parameters
    ----------
    config : Mapping[str, Any]
        Configuration dict / YAML that may contain a covariance entry
        under `cov_name`. The covariance can be:
          - (N, N) array-like full matrix,
          - (N,) array-like interpreted as diagonal,
          - scalar (applied to the diagonal).
    state_dim : int | None, optional (default: None)
        If provided, fixes the number of DoF (N). If None, the filter tries to
        infer N from `config[cov_name]` (when present). If still unknown, N is
        lazily inferred from the first vector passed to `filter_data`, or from
        the first call to `set_offsets`.
    smooth_factor : float, optional (default: 0.1)
        Scales the *process* noise covariance Q as
        `Q = smooth_factor * measurement_covariance`.
        Smaller values → heavier smoothing (slower response). Must be ≥ 0.
    cov_name : str | None, optional (default: None)
        Key in `config` that stores the measurement covariance. If missing or
        None, an identity covariance (scaled by `default_cov_val`) is used.
    default_cov_val : float, optional (default: 1e-2)
        Variance value used for the diagonal when a covariance is not provided.
    dtype : np.dtype, optional (default: np.float32)
        Floating dtype for internal arrays.

    Notes
    -----
    - If you change the number of DoF, you should either create a new
      `DataFilter` or call `reset(hard=True)` and then provide a new first
      measurement (or offsets) with the new size so the filter can re‑infer N.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        state_dim: Optional[int] = None,
        smooth_factor: float = 0.1,
        cov_name: Optional[str] = None,
        default_cov_val: float = 1e-2,
        dtype: np.dtype = np.float32,
    ) -> None:
        if smooth_factor < 0:
            raise ValueError("smooth_factor must be non-negative.")

        self._dtype: np.dtype = dtype
        self.smooth_factor: float = float(smooth_factor)
        self.default_cov_val: float = float(default_cov_val)

        # Store raw covariance (may be None / 1D / 2D / scalar) and try to infer N.
        self._pending_cov_like: Any = None
        if cov_name is not None:
            self._pending_cov_like = config.get(cov_name, None)

        inferred_from_cov = self._infer_dim_from_cov(self._pending_cov_like)
        if state_dim is None:
            state_dim = inferred_from_cov

        # Core state (initialized once the dimension is known)
        self.state_dim: Optional[int] = None
        self.offsets: Optional[np.ndarray] = None
        self.prev_estimate: Optional[np.ndarray] = None
        self.measurement_covariance: Optional[np.ndarray] = None
        self.estimate_covariance: Optional[np.ndarray] = None
        self.process_covariance: Optional[np.ndarray] = None
        self._I: Optional[np.ndarray] = None

        # Whether a meaningful estimate has been produced (after first update)
        self.initialized: bool = False

        # If we already know the dimension, finalize all matrices now.
        if state_dim is not None:
            self._setup_dimension(int(state_dim))
            self._finalize_covariances()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_offsets(self, offset_array: Sequence[Number]) -> None:
        """
        Set static offsets (bias) to be subtracted from incoming measurements.

        If the filter dimension is not yet known, this call **fixes** it to
        the length of `offset_array` and triggers internal initialization.

        Parameters
        ----------
        offset_array : Sequence[float]
            Shape must be (N,), where N is the number of DoF.
        """
        offset_arr = np.asarray(offset_array, dtype=self._dtype).reshape(-1)
        if self.state_dim is None:
            self._setup_dimension(offset_arr.size)
            self._finalize_covariances()
        elif offset_arr.shape != (self.state_dim,):
            raise ValueError(
                f"offset_array must have shape ({self.state_dim},), "
                f"got {offset_arr.shape}."
            )
        self.offsets = offset_arr

    def filter_data(self, input_array: Sequence[Number]) -> np.ndarray:
        """
        Apply offsets and Kalman smoothing to a single input vector.

        If the dimension was not yet known, this call infers it from the input
        length and finalizes the internal matrices.

        Parameters
        ----------
        input_array : Sequence[float]
            Raw measurement with length = N.

        Returns
        -------
        np.ndarray
            Smoothed vector of shape (N,).
        """
        z = np.asarray(input_array, dtype=self._dtype).reshape(-1)

        # Lazily initialize dimension and covariances if needed.
        if self.state_dim is None:
            self._setup_dimension(z.size)
            self._finalize_covariances()

        if z.shape != (self.state_dim,):
            raise ValueError(
                f"input_array must have shape ({self.state_dim},), "
                f"got {z.shape}."
            )

        # Ensure offsets exist (default to zeros when not set explicitly).
        if self.offsets is None:
            self.offsets = np.zeros(self.state_dim, dtype=self._dtype)

        # Apply offsets
        z_adj = z - self.offsets

        # First measurement → initialize estimate without a prediction step.
        if not self.initialized or self.prev_estimate is None:
            self.prev_estimate = z_adj.copy()
            self.initialized = True
            return z_adj

        return self._kalman_step(z_adj)

    def reset(self, *, hard: bool = False) -> None:
        """
        Reset the filter's internal state.

        Parameters
        ----------
        hard : bool, optional (default: False)
            If False: keep dimension and covariances; only clear the running
            estimate so the next `filter_data` call will treat its input as the
            first measurement.
            If True: also forget the dimension and all matrices, so the next
            call can re-infer N and rebuild everything lazily.
        """
        self.initialized = False

        if not hard:
            if self.state_dim is None:
                return  # Nothing else to do.
            # Keep dimension and covariances; clear only the dynamic parts.
            self.prev_estimate = np.zeros(self.state_dim, dtype=self._dtype)
            if self.measurement_covariance is not None:
                self.estimate_covariance = self.measurement_covariance.copy()
            return

        # Hard reset: forget dimension and all matrices
        self.state_dim = None
        self.prev_estimate = None
        self.offsets = None
        self.measurement_covariance = None
        self.estimate_covariance = None
        self.process_covariance = None
        self._I = None
        # Keep the pending covariance (user config) so we can reuse it on re‑init.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_dimension(self, n: int) -> None:
        if n <= 0:
            raise ValueError("state_dim must be a positive integer.")
        self.state_dim = int(n)
        self.offsets = np.zeros(n, dtype=self._dtype)
        self.prev_estimate = np.zeros(n, dtype=self._dtype)
        self._I = np.eye(n, dtype=self._dtype)

    def _finalize_covariances(self) -> None:
        """Build measurement, estimate, and process covariances."""
        assert self.state_dim is not None, "Dimension must be set first."
        self.measurement_covariance = self._normalize_covariance(
            self._pending_cov_like, self.state_dim, self.default_cov_val, self._dtype
        )
        # Start with the same scale as R (typical basic choice)
        self.estimate_covariance = self.measurement_covariance.copy()
        self.process_covariance = self.measurement_covariance * self.smooth_factor

    @staticmethod
    def _infer_dim_from_cov(cov_like: Any) -> Optional[int]:
        """
        Try to infer N from a covariance-like object.
        Returns N or None if it cannot be inferred.
        """
        if cov_like is None:
            return None
        cov = np.asarray(cov_like)
        if cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
            return int(cov.shape[0])
        if cov.ndim == 1:
            return int(cov.shape[0])
        if np.isscalar(cov):
            return None  # scalar does not encode dimension
        return None

    @staticmethod
    def _normalize_covariance(
        cov_like: Any,
        n: int,
        default_diag: float,
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        Coerce various covariance representations into an (n, n) float array.
        Accepted forms:
          - None → default_diag * I
          - scalar s → s * I
          - (n,) → diag(cov_like)
          - (n, n) → used as is (symmetrized for safety)
        """
        if cov_like is None:
            return (np.eye(n, dtype=dtype) * default_diag).astype(dtype, copy=False)

        cov = np.asarray(cov_like, dtype=dtype)
        if np.isscalar(cov) or cov.ndim == 0:
            return (np.eye(n, dtype=dtype) * float(cov)).astype(dtype, copy=False)

        if cov.ndim == 1:
            if cov.shape[0] != n:
                raise ValueError(
                    f"1D covariance length {cov.shape[0]} does not match n={n}."
                )
            return np.diag(cov.astype(dtype, copy=False))

        if cov.ndim == 2:
            if cov.shape != (n, n):
                raise ValueError(
                    f"2D covariance shape {cov.shape} does not match (n, n)=({n}, {n})."
                )
            # Symmetrize lightly to reduce numerical asymmetry
            return ((cov + cov.T) * 0.5).astype(dtype, copy=False)

        raise ValueError("Unsupported covariance format.")

    # ------------------------------------------------------------------
    # Internal: one Kalman iteration (identity model)
    # ------------------------------------------------------------------
    def _kalman_step(self, measurement: np.ndarray) -> np.ndarray:
        assert self.state_dim is not None
        assert self.prev_estimate is not None
        assert self.estimate_covariance is not None
        assert self.process_covariance is not None
        assert self.measurement_covariance is not None

        # Prediction (identity state transition)
        pred_est = self.prev_estimate
        pred_cov = self.estimate_covariance + self.process_covariance

        # Innovation
        innovation = measurement - pred_est
        innov_cov = pred_cov + self.measurement_covariance  # S = P + R

        # Kalman gain K = P_pred * S^{-1}
        # Use solve on transposes to avoid explicit inverse on the right.
        try:
            K = np.linalg.solve(innov_cov.T, pred_cov.T).T
        except np.linalg.LinAlgError:
            # Graceful fallback: pseudo-inverse if S is singular/ill‑conditioned
            K = pred_cov @ np.linalg.pinv(innov_cov)

        # Update
        current_est = pred_est + K @ innovation
        # Joseph form omitted for speed
        self.estimate_covariance = (self._I - K) @ pred_cov
        self.prev_estimate = current_est

        return current_est
