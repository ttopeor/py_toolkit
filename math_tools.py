import numpy as np
from scipy.spatial.transform import Rotation as R


def invert_position(position):
    position_matrix = position_to_trans_matrix(position)
    position_matrix_inv = np.linalg.inv(position_matrix)
    position_inv = trans_matrix_to_position(position_matrix_inv)
    return position_inv


def position_to_trans_matrix(transform):
    """
    Converts position [x, y, z] and Euler angles [roll, pitch, yaw] (in radians)
    into a 4x4 homogeneous transformation matrix T_AB.

    Here, 'transform' describes the pose of frame B relative to frame A.
    i.e., T_AB transforms coordinates from A to B.

    :param transform: [x, y, z, roll, pitch, yaw]
    :return: a 4x4 numpy array, T_AB
    """
    if len(transform) != 6:
        raise ValueError(
            "Transform must have 6 elements: [x, y, z, roll, pitch, yaw].")

    x, y, z, roll, pitch, yaw = transform

    # Rotation from Euler angles (in radians, 'XYZ' convention)
    rot = R.from_euler('XYZ', [roll, pitch, yaw], degrees=False)
    R_mat = rot.as_matrix()  # 3x3

    # Build the 4x4 homogeneous transform
    T_AB = np.eye(4)
    T_AB[0:3, 0:3] = R_mat
    T_AB[0:3, 3] = [x, y, z]

    return T_AB


def trans_matrix_to_position(T_AB):
    """
    Converts a 4x4 homogeneous transformation matrix back into
    position [x, y, z] and Euler angles [roll, pitch, yaw] (in radians).

    :param T_AB: 4x4 numpy array
    :return: [x, y, z, roll, pitch, yaw]
    """
    if T_AB.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4.")

    # Extract translation
    x, y, z = T_AB[0:3, 3]

    # Extract rotation part and convert to Euler angles
    R_mat = T_AB[0:3, 0:3]
    rot = R.from_matrix(R_mat)
    roll, pitch, yaw = rot.as_euler('XYZ', degrees=False)

    return np.array([x, y, z, roll, pitch, yaw])


def parent_to_child(parent_cartesian_position, delta_position):
    """
    Given a parent frame pose and a relative delta pose, compute
    the resulting child frame pose.

    :param parent_cartesian_position: [x, y, z, roll, pitch, yaw] of the parent frame
    :param delta_position: [x, y, z, roll, pitch, yaw] relative transform from the parent frame
    :return: [x, y, z, roll, pitch, yaw] of the child frame
    """
    # 1. Convert input vectors to 4x4 transforms
    ParentPositionMatrix = position_to_trans_matrix(parent_cartesian_position)
    DeltaPositionMatrix = position_to_trans_matrix(delta_position)

    # 2. Multiply to get the child's transform matrix
    ChildPositionMatrix = np.dot(ParentPositionMatrix, DeltaPositionMatrix)

    # 3. Convert back to [x, y, z, roll, pitch, yaw]
    ChildPosition = trans_matrix_to_position(ChildPositionMatrix)

    return ChildPosition


def child_to_parent(child_cartesian_position, delta_position):
    """
    Computes the parent's pose in the world frame, given:
      - child's pose in the world frame (child_cartesian_position)
      - the transform from the parent to the child (delta_position).

    Go counterpart for reference:
        ChildPositionMatrix := PositionToTransMatrix(childCartesianPosition)
        DeltaPositionMatrix := PositionToTransMatrix(deltaPosition)

        invDeltaPositionMatrix := Inverse(DeltaPositionMatrix)
        ParentPositionMatrix := ChildPositionMatrix * invDeltaPositionMatrix
        ParentPosition := MatrixToPosition(ParentPositionMatrix)
    """
    # 1. Convert to 4x4 homogeneous transformation matrices
    child_position_matrix = position_to_trans_matrix(child_cartesian_position)
    delta_position_matrix = position_to_trans_matrix(delta_position)

    # 2. Invert the delta_position_matrix
    inv_delta_position_matrix = np.linalg.inv(delta_position_matrix)

    # 3. Compute the parent's matrix
    parent_position_matrix = np.dot(
        child_position_matrix, inv_delta_position_matrix)

    # 4. Convert back to [x, y, z, roll, pitch, yaw]
    parent_position = trans_matrix_to_position(parent_position_matrix)

    return parent_position


def calculate_delta_position(parent_cartesian_position, child_cartesian_position):
    """
    Computes the transform from the parent frame to the child frame (delta_position),
    given:
      - parent's pose in the world frame (parent_cartesian_position)
      - child's pose in the world frame (child_cartesian_position).

    Go counterpart for reference:
        parentPositionMatrix := PositionToTransMatrix(parentCartesianPosition)
        childPositionMatrix := PositionToTransMatrix(childCartesianPosition)

        invParentPositionMatrix := Inverse(parentPositionMatrix)
        deltaPositionMatrix := invParentPositionMatrix * childPositionMatrix
        deltaPosition := MatrixToPosition(deltaPositionMatrix)
    """
    # 1. Convert to 4x4 homogeneous transformation matrices
    parent_position_matrix = position_to_trans_matrix(
        parent_cartesian_position)
    child_position_matrix = position_to_trans_matrix(child_cartesian_position)

    # 2. Invert the parent_position_matrix
    inv_parent_position_matrix = np.linalg.inv(parent_position_matrix)

    # 3. Compute the matrix that transforms from parent to child
    delta_position_matrix = np.dot(
        inv_parent_position_matrix, child_position_matrix)

    # 4. Convert back to [x, y, z, roll, pitch, yaw]
    delta_position = trans_matrix_to_position(delta_position_matrix)

    return delta_position


def compute_force_at_B(measured_force_moment_A, transform):
    """
    Given a wrench measured at point A (in A-coords), find the equivalent wrench
    at point B (expressed in B-coords).

    :param measured_force_moment_A: [Fx_A, Fy_A, Fz_A, Mx_A, My_A, Mz_A],
                                    measured about point A, in A's coordinate system.
    :param transform: [x, y, z, roll, pitch, yaw],
                      describing the transformation from A to B.
                      That is, T_AB transforms vectors from A-coords to B-coords,
                      and [x,y,z] is the position of B in A-coords.
    :return: [Fx_B, Fy_B, Fz_B, Mx_B, My_B, Mz_B],
             i.e. the force & torque about B, in B-coords.
    """
    # Build the homogeneous transformation from A to B
    T_AB = position_to_trans_matrix(transform)

    # Extract rotation and translation
    R_AB = T_AB[0:3, 0:3]   # from A-coords -> B-coords
    r_AB = T_AB[0:3, 3]     # position of B w.r.t. A, in A-coords

    # Convert input to numpy arrays
    measured_force_moment_A = np.asarray(
        measured_force_moment_A, dtype=float).reshape(6,)
    F_A = measured_force_moment_A[0:3]  # Force at A (A-coords)
    M_A = measured_force_moment_A[3:6]  # Torque about A (A-coords)

    # 1) Force in B-coords
    F_B = R_AB @ F_A

    # 2) Torque about B, in B-coords
    #    M_B = R_AB [ M_A - ( r_AB x F_A ) ]
    M_B = R_AB @ (M_A - np.cross(r_AB, F_A))

    # Combine
    force_moment_B = np.concatenate([F_B, M_B])
    return force_moment_B.tolist()


def quantize_to_resolution(value, resolution):

    sign = 1 if value >= 0 else -1

    x = abs(value)

    multiple = int(x // resolution)
    base = multiple * resolution
    remainder = x - base

    if remainder >= 0.5:
        quantized = base + resolution
    else:
        quantized = base

    return sign * quantized


def unwrap_angle(angle_now, angle_prev):

    delta = angle_now - (angle_prev % (2*np.pi))
    while delta < -np.pi:
        delta += 2*np.pi
    while delta > np.pi:
        delta -= 2*np.pi

    return angle_prev + delta


def unwrap_rpy(new, old):

    new = np.array(new)
    old = np.array(old)
    result = new.copy()
    for i in [3, 4, 5]:  # roll, pitch, yaw
        delta = new[i] - old[i]
        if delta > np.pi:
            result[i] -= 2 * np.pi
        elif delta < -np.pi:
            result[i] += 2 * np.pi
    return result


def represent_wrench_to_B(wrench_a, transform_6d):

    T_AB = position_to_trans_matrix(transform_6d)
    R_AB = T_AB[:3, :3]
    t_AB = T_AB[:3, 3]

    f_a = np.array(wrench_a[:3])
    tau_a = np.array(wrench_a[3:])

    cross_term = np.cross(t_AB, tau_a)

    f_b = R_AB.T @ f_a - R_AB.T @ cross_term
    tau_b = R_AB.T @ tau_a

    wrench_b = np.concatenate([f_b, tau_b])
    return wrench_b


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def rotmat_to_rot6d(Rm: np.ndarray) -> np.ndarray:
    """(…,3,3) → (…,6) : take first two columns"""
    return Rm[..., :3, :2].reshape(*Rm.shape[:-2], 6, order="F").astype(np.float32)


def rot6d_to_rotmat(d6: np.ndarray) -> np.ndarray:

    d6 = np.asarray(d6, dtype=np.float32)
    a1a2 = d6.reshape(*d6.shape[:-1], 3, 2, order="F")   # (...,3,2)
    a1 = a1a2[..., 0]                                    # (...,3)
    a2 = a1a2[..., 1]

    b1 = normalize(a1)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(a2 - proj)
    b3 = np.cross(b1, b2)

    rot = np.stack((b1, b2, b3), axis=-1)                # (...,3,3)
    return rot.astype(np.float32)


def pose6_to_pose9(pose6: np.ndarray) -> np.ndarray:
    """
    [x,y,z, roll,pitch,yaw]  ->  [x,y,z, rot6d]
    pose6 shape (..., 6) ; returns same batch shape (..., 9)
    """
    pose6 = np.asarray(pose6, dtype=np.float32)
    pos, rpy = pose6[..., :3], pose6[..., 3:]
    rotmat = R.from_euler('XYZ', rpy, degrees=False).as_matrix()  # (...,3,3)
    rot6d = rotmat_to_rot6d(rotmat)
    return np.concatenate((pos, rot6d), axis=-1).astype(np.float32)


def pose9_to_pose6(pose9: np.ndarray) -> np.ndarray:
    """
    [x,y,z, rot6d]  ->  [x,y,z, roll,pitch,yaw]
    pose9 shape (..., 9) ; returns same batch shape (..., 6)
    """
    pose9 = np.asarray(pose9, dtype=np.float32)
    pos, d6 = pose9[..., :3], pose9[..., 3:]
    rotmat = rot6d_to_rotmat(d6)                                  # (...,3,3)
    rpy = R.from_matrix(rotmat).as_euler('XYZ', degrees=False)    # (...,3)
    return np.concatenate((pos, rpy), axis=-1).astype(np.float32)


def random_perturb_and_apply(
    pose,                        # [x, y, z, roll, pitch, yaw]
    *,
    trans_mu=0.0,                # mean for translation noise
    trans_sigma=0.01,           # stddev for translation noise (m)
    rot_mu=0.0,                  # mean for rotation noise
    rot_sigma=np.deg2rad(2.5),   # stddev for rotation noise (rad)
    trans_bounds=(-0.03, 0.03),  # (min, max) bounds for translation noise (m)
    # (min, max) bounds for rotation noise (rad)
    rot_bounds=(-np.deg2rad(7), np.deg2rad(7))
):
    """
    Add Gaussian noise to [x, y, z, roll, pitch, yaw] with specified bounds,
    then apply the delta using parent_to_child.

    Returns
    -------
    final_pose : list
        Pose after applying noise, format [x, y, z, roll, pitch, yaw].
    delta : tuple
        The applied noise [dx, dy, dz, droll, dpitch, dyaw].
    """
    # Ensure inputs are arrays of length 3
    def to3(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) != 3:
                raise ValueError("Expected length=3 for per-axis value")
            return np.array(v, dtype=float)
        return np.full(3, float(v))

    mu_t, sig_t = to3(trans_mu), to3(trans_sigma)
    mu_r, sig_r = to3(rot_mu), to3(rot_sigma)

    lo_t, hi_t = float(trans_bounds[0]), float(trans_bounds[1])
    lo_r, hi_r = float(rot_bounds[0]),  float(rot_bounds[1])

    # Sample translation noise
    trans_noise = np.random.normal(mu_t, sig_t)
    trans_noise = np.clip(trans_noise, lo_t, hi_t)

    # Sample rotation noise
    rot_noise = np.random.normal(mu_r, sig_r)
    rot_noise = np.clip(rot_noise, lo_r, hi_r)

    delta = tuple(trans_noise.tolist() + rot_noise.tolist())

    parent_pose = tuple(float(x) for x in pose)
    final_pose = parent_to_child(parent_pose, delta)

    # Ensure final_pose is list
    final_pose = list(final_pose)
    final_pose = [float(x) for x in final_pose]

    return final_pose, delta


def print_array(name: str, a: np.ndarray) -> str:

    contain = np.array2string(
        a,
        precision=6,
        suppress_small=True,
        floatmode="fixed",
        max_line_width=120,
    )
    print(name + ": ", contain)


if __name__ == "__main__":

    wrenchA = [0, 0, 0, 0.0, 0.0, 0.2669428476]
    transform_BA = [0, 0, 0, np.pi, 0.0, np.pi/2]

    wrenchA_B = represent_wrench_to_B(wrenchA, transform_BA)
    print("Force & Moment at B (in B-frame):", wrenchA_B)
