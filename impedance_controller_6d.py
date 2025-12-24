import numpy as np


class ImpedanceController6D:
    """A simple 6‑DoF **impedance controller** implemented at the **position level**.

    Each Cartesian degree of freedom (x, y, z, roll, pitch, yaw) is treated as an
    independent virtual mass–spring–damper system.  The controller is able to
    *degrade* to a first‑order (mass‑less) model for any axis where `M[i] == 0`.

    Parameters
    ----------
    M, B, K : Sequence[float] of length 6
        Diagonal elements of the virtual **mass**, **damping** and **stiffness**
        matrices, i.e. ``[M_x, M_y, M_z, M_roll, M_pitch, M_yaw]``.
        If ``M[i] == 0`` the corresponding channel behaves as a first‑order
        impedance: ``B_i · e_v + K_i · e_p = F_ext``.
    """

    def __init__(self, M, B, K):
        assert len(M) == 6 and len(B) == 6 and len(
            K) == 6, "M, B and K must have length 6"

        self.M_diag = np.array(M, dtype=float)
        self.B_diag = np.array(B, dtype=float)
        self.K_diag = np.array(K, dtype=float)

        # Internal state (controller‑side estimate)
        self.p: np.ndarray = np.zeros(6)  # position [x y z roll pitch yaw]
        self.v: np.ndarray = np.zeros(6)  # velocity

        # Desired (reference) motion — can be changed online
        self.p_d: np.ndarray = np.zeros(6)
        self.v_d: np.ndarray = np.zeros(6)
        self.a_d: np.ndarray = np.zeros(6)

    # ---------------------------------------------------------------------
    # Public setters
    # ---------------------------------------------------------------------
    def set_desired_state(self, p_d, v_d=None, a_d=None):
        """Update the desired pose / velocity / acceleration.

        Parameters
        ----------
        p_d : Sequence[float]
            Desired pose ``[x, y, z, roll, pitch, yaw]``.
        v_d, a_d : Sequence[float] | None
            Optional desired velocity and acceleration. If omitted they keep
            their previous values (default is all‑zeros).
        """
        self.p_d = np.array(p_d, dtype=float)
        if v_d is not None:
            self.v_d = np.array(v_d, dtype=float)
        if a_d is not None:
            self.a_d = np.array(a_d, dtype=float)

    def initialize_state(self, p_init, v_init=None):
        """Synchronise the controller with the robot's **current** pose.

        Parameters
        ----------
        p_init : Sequence[float]
            Current end‑effector pose.
        v_init : Sequence[float] | None, optional
            Current end‑effector velocity.  If *None* the velocity is reset to
            zero.
        """
        self.p = np.array(p_init, dtype=float)
        if v_init is not None:
            self.v = np.array(v_init, dtype=float)
        else:
            self.v = np.zeros(6)

    # ---------------------------------------------------------------------
    # Main update step
    # ---------------------------------------------------------------------
    def update(self, F_ext_6d, dt):
        """Advance the controller by one time‑step.

        Parameters
        ----------
        F_ext_6d : Sequence[float]
            External wrench ``[Fx, Fy, Fz, Mx, My, Mz]`` measured at the TCP.
        dt : float
            Integration step in **seconds**.

        Returns
        -------
        np.ndarray
            The new pose `p_new` that should be commanded to the robot for this
            control tick.
        """
        F_ext_6d = np.array(F_ext_6d, dtype=float)

        # Position and velocity errors (actual – desired)
        e_p = self.p - self.p_d
        e_v = self.v - self.v_d

        p_new = self.p.copy()
        v_new = self.v.copy()

        # Loop over the 6 Cartesian channels independently
        for i in range(6):
            M_i, B_i, K_i = self.M_diag[i], self.B_diag[i], self.K_diag[i]
            F_i = F_ext_6d[i]
            ep_i, ev_i, ad_i = e_p[i], e_v[i], self.a_d[i]

            if abs(M_i) > 1e-9:
                # -------------------- second‑order impedance (M_i ≠ 0) --------------------
                #   p_ddot = (F_i − B_i·e_v − K_i·e_p) / M_i + a_d
                a_i = (F_i - B_i * ev_i - K_i * ep_i) / M_i + ad_i
                v_new[i] += a_i * dt           # integrate acceleration
                p_new[i] += v_new[i] * dt      # integrate velocity

            else:
                # --------------------- first‑order impedance (M_i = 0) ---------------------
                if abs(B_i) < 1e-9:
                    # Special case: B = 0  ⇒  K_i·e_p = F_i
                    if abs(K_i) > 1e-9:
                        p_new[i] = self.p_d[i] + F_i / K_i
                        # pseudo‑velocity for continuity
                        v_new[i] = (p_new[i] - self.p[i]) / dt
                    # Else B = K = 0 → ill‑defined; keep previous state
                else:
                    ev_i_new = (F_i - K_i * ep_i) / B_i
                    v_new[i] = self.v_d[i] + ev_i_new
                    p_new[i] += v_new[i] * dt

        # Commit the updated state
        self.p = p_new
        self.v = v_new
        return p_new

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def reset(self):
        """Reset all internal and desired states to zero."""
        self.p[:] = 0.0
        self.v[:] = 0.0
        self.p_d[:] = 0.0
        self.v_d[:] = 0.0
        self.a_d[:] = 0.0
