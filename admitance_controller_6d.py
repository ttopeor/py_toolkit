import numpy as np


class AdmittanceController6D:
    """A 6-DoF admittance controller.
    """

    def __init__(self, M, B, K):
        """Create a new controller instance.

        Parameters
        ----------
        M : sequence of float (len = 6)
            Virtual masses ``[m_x, m_y, m_z, m_roll, m_pitch, m_yaw]``.
        B : sequence of float (len = 6)
            Virtual damping coefficients.
        K : sequence of float (len = 6)
            Virtual stiffness coefficients.
        """
        assert len(M) == 6 and len(B) == 6 and len(
            K) == 6, "Lengths of M, B, and K must be 6"

        # Convert to NumPy arrays for vectorised maths
        self.M = np.array(M, dtype=float)
        self.B = np.array(B, dtype=float)
        self.K = np.array(K, dtype=float)

        # State variables
        # Current pose (x, y, z, rx, ry, rz)
        self.position = np.zeros(6, dtype=float)
        # Current velocity in each DoF
        self.velocity = np.zeros(6, dtype=float)
        self.desired_position = np.zeros(6, dtype=float)  # Reference pose
        # Reference external wrench
        self.desired_force = np.zeros(6, dtype=float)

    # ---------------------------------------------------------------------
    # Set‑up helpers
    # ---------------------------------------------------------------------
    def set_desired_force(self, desired_force_6d):
        """Set the desired external wrench (Nx, Ny, Nz, Tx, Ty, Tz)."""
        assert len(desired_force_6d) == 6, "Desired force must have length 6"
        self.desired_force = np.array(desired_force_6d, dtype=float)

    def set_desired_position(self, desired_pos_6d):
        """Set the desired pose (position + orientation)."""
        assert len(desired_pos_6d) == 6, "Desired position must have length 6"
        self.desired_position = np.array(desired_pos_6d, dtype=float)

    def set_current_position(self, pos_6d):
        """Update the controller's internal position estimate."""
        assert len(pos_6d) == 6, "Current position must have length 6"
        self.position = np.array(pos_6d, dtype=float)

    def initialize_state(self, position_init, velocity_init=None):
        """Initialise the internal state of the controller.

        Parameters
        ----------
        position_init : sequence of float (len = 6)
            Initial pose to store in :pyattr:`position`.
        velocity_init : sequence of float (len = 6), optional
            Initial velocity. If *None*, the velocity vector is zeroed.
        """
        assert len(position_init) == 6, "position_init must have length 6"
        self.position = np.array(position_init, dtype=float)

        if velocity_init is not None:
            assert len(velocity_init) == 6, "velocity_init must have length 6"
            self.velocity = np.array(velocity_init, dtype=float)
        else:
            self.velocity = np.zeros(6, dtype=float)

    # ---------------------------------------------------------------------
    # Core update step
    # ---------------------------------------------------------------------
    def update(self, measured_force_6d, dt):
        """Compute a pose increment given the measured external wrench.

        Parameters
        ----------
        measured_force_6d : sequence of float (len = 6)
            Current external wrench from the force/torque sensor.
        dt : float
            Control period [s].

        Returns
        -------
        numpy.ndarray (shape = (6,))
            Pose increment \u0394x to be applied.
        """
        measured_force_6d = np.array(measured_force_6d, dtype=float)

        # Force error (actual – desired)
        force_error_6d = measured_force_6d - self.desired_force

        new_velocity = np.zeros(6, dtype=float)
        delta_x = np.zeros(6, dtype=float)

        for i in range(6):
            # Governing equation: M x_ddot + B x_dot + K (x - x_ref) = F_err
            if abs(self.M[i]) < 1e-9:  # Degenerate: ignore acceleration term
                if abs(self.B[i]) < 1e-9:
                    # Nearly rigid – command no motion
                    x_dot_i = 0.0
                else:
                    # First‑order system (damping + stiffness)
                    x_dot_i = (
                        force_error_6d[i]
                        - self.K[i] * (self.position[i] -
                                       self.desired_position[i])
                    ) / self.B[i]
            else:
                # Full second‑order dynamics
                x_ddot = (
                    force_error_6d[i]
                    - self.B[i] * self.velocity[i]
                    - self.K[i] * (self.position[i] - self.desired_position[i])
                ) / self.M[i]
                x_dot_i = self.velocity[i] + x_ddot * dt

            new_velocity[i] = x_dot_i
            delta_x[i] = x_dot_i * dt

        # Update internal velocity state; position is left to the caller
        self.velocity = new_velocity
        return delta_x

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def reset(self):
        """Reset all internal state variables to zero."""
        self.position[:] = 0.0
        self.velocity[:] = 0.0
        self.desired_position[:] = 0.0
        self.desired_force[:] = 0.0
