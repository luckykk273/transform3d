import numpy as np
import pytransform3d.rotations as pr
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import warnings
from lib_wrapper import LibWrapper

libtrans = LibWrapper()


def test_norm_vector():
    """Test normalization of vectors."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = pr.random_vector(rng, 3)
        u = libtrans.normalize_vector(v)
        assert pytest.approx(np.linalg.norm(u)) == 1


def test_angle_between_vectors():
    """Test function to compute angle between two vectors."""
    v = np.array([1, 0, 0])
    a = np.array([0, 1, 0, np.pi / 2])
    R = libtrans.axang_to_rmat(a)
    vR = np.dot(R, v)
    assert pytest.approx(libtrans.angle_between_vectors(vR, v)) == a[-1]
    v = np.array([0, 1, 0])
    a = np.array([1, 0, 0, np.pi / 2])
    R = libtrans.axang_to_rmat(a)
    vR = np.dot(R, v)
    assert pytest.approx(libtrans.angle_between_vectors(vR, v)) == a[-1]
    v = np.array([0, 0, 1])
    a = np.array([1, 0, 0, np.pi / 2])
    R = libtrans.axang_to_rmat(a)
    vR = np.dot(R, v)
    assert pytest.approx(libtrans.angle_between_vectors(vR, v)) == a[-1]


def test_angle_between_close_vectors():
    """Test angle between close vectors.

    See issue #47.
    """
    a = np.array([0.9689124217106448, 0.24740395925452294, 0.0, 0.0])
    b = np.array([0.9689124217106448, 0.247403959254523, 0.0, 0.0])
    angle = libtrans.angle_between_vectors(a, b, n=4)
    assert pytest.approx(angle) == 0.0


def test_angle_to_zero_vector_is_nan():
    """Test angle to zero vector."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    angle = libtrans.angle_between_vectors(a, b, n=2)
    # NOTE: Because it is the C library wrapper, 
    #       it doesn't raise any warnings.
    # with warnings.catch_warnings(record=True) as w:
    #     angle = libtrans.angle_between_vectors(a, b, n=2)
    #     assert len(w) == 1
    assert np.isnan(angle)


def test_norm_zero_vector():
    """Test normalization of zero vector."""
    normalized = libtrans.normalize_vector(np.zeros(3))
    assert np.isfinite(np.linalg.norm(normalized))


def test_norm_angle():
    """Test normalization of angle."""
    rng = np.random.default_rng(0)
    a_norm = rng.uniform(-np.pi, np.pi, size=(100,))
    for b in np.linspace(-10.0 * np.pi, 10.0 * np.pi, 11):
        a = a_norm + b
        assert_array_almost_equal(
            np.array([libtrans.normalize_ang(a_in) for a_in in a], dtype=np.double),
            a_norm
        )

    assert pytest.approx(libtrans.normalize_ang(-np.pi)) == np.pi
    assert pytest.approx(libtrans.normalize_ang(np.pi)) == np.pi


def test_norm_axis_angle():
    """Test normalization of angle-axis representation."""
    a = np.array([1.0, 0.0, 0.0, np.pi])
    n = libtrans.normalize_axang(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    n = libtrans.normalize_axang(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    n = libtrans.normalize_axang(a)
    assert_array_almost_equal(a, n)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        angle = rng.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a[3] = angle
        n = libtrans.normalize_axang(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a[3] = angle + angle_offset
            n2 = libtrans.normalize_axang(a)
            assert_array_almost_equal(n, n2)


def test_norm_compact_axis_angle():
    """Test normalization of compact angle-axis representation."""
    a = np.array([np.pi, 0.0, 0.0])
    n = libtrans.normalize_compact_axang(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, np.pi, 0.0])
    n = libtrans.normalize_compact_axang(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, np.pi])
    n = libtrans.normalize_compact_axang(a)
    assert_array_almost_equal(a, n)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_compact_axis_angle(rng)
        axis = a / np.linalg.norm(a)
        angle = rng.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a = axis * angle
        n = libtrans.normalize_compact_axang(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a = axis * (angle + angle_offset)
            n2 = libtrans.normalize_compact_axang(a)
            assert_array_almost_equal(n, n2)


def test_vector_projection_on_zero_vector():
    """Test projection on zero vector."""
    rng = np.random.default_rng(23)
    for _ in range(5):
        a = pr.random_vector(rng, 3)
        a_on_b = libtrans.orthogonal_project(a, np.zeros(3))
        assert_array_almost_equal(a_on_b, np.zeros(3))


def test_vector_projection():
    """Test orthogonal projection of one vector to another vector."""
    a = np.ones(3)
    a_on_unitx = libtrans.orthogonal_project(a, pr.unitx)
    assert_array_almost_equal(a_on_unitx, pr.unitx)
    assert pytest.approx(
        libtrans.angle_between_vectors(a_on_unitx, pr.unitx)) == 0.0

    a2_on_unitx = libtrans.orthogonal_project(2 * a, pr.unitx)
    assert_array_almost_equal(a2_on_unitx, 2 * pr.unitx)
    assert pytest.approx(
        libtrans.angle_between_vectors(a2_on_unitx, pr.unitx)) == 0.0
    
    a_on_unity = libtrans.orthogonal_project(a, pr.unity)
    assert_array_almost_equal(a_on_unity, pr.unity)
    assert pytest.approx(
        libtrans.angle_between_vectors(a_on_unity, pr.unity)) == 0.0

    minus_a_on_unity = libtrans.orthogonal_project(-a, pr.unity)
    assert_array_almost_equal(minus_a_on_unity, -pr.unity)
    assert pytest.approx(
        libtrans.angle_between_vectors(minus_a_on_unity, pr.unity)) == np.pi

    a_on_unitz = libtrans.orthogonal_project(a, pr.unitz)
    assert_array_almost_equal(a_on_unitz, pr.unitz)
    assert pytest.approx(
        libtrans.angle_between_vectors(a_on_unitz, pr.unitz)) == 0.0

    unitz_on_a = libtrans.orthogonal_project(pr.unitz, a)
    assert_array_almost_equal(unitz_on_a, np.ones(3) / 3.0)
    assert pytest.approx(libtrans.angle_between_vectors(unitz_on_a, a)) == 0.0

    unitx_on_unitx = libtrans.orthogonal_project(pr.unitx, pr.unitx)
    assert_array_almost_equal(unitx_on_unitx, pr.unitx)
    assert pytest.approx(
        libtrans.angle_between_vectors(unitx_on_unitx, pr.unitx)) == 0.0


def test_check_matrix():
    """Test input validation for rotation matrix."""
    R = np.eye(3)
    assert libtrans.is_rmat(R)

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0.1, 1]])
    assert not libtrans.is_rmat(R)

    R = np.array([[1, 0, 1e-16], [0, 1, 0], [0, 0, 1]])
    assert libtrans.is_rmat(R)

    R = -np.eye(3)
    assert not libtrans.is_rmat(R)


def test_check_axis_angle():
    """Test input validation for axis-angle representation."""
    a_list = [1, 0, 0, 0]
    a = libtrans.normalize_axang(np.array(a_list))
    assert_array_almost_equal(a_list, a)

    rng = np.random.default_rng(0)
    a = np.empty(4)
    a[:3] = pr.random_vector(rng, 3)
    a[3] = rng.standard_normal() * 4.0 * np.pi
    a2 = libtrans.normalize_axang(a)
    pr.assert_axis_angle_equal(a, a2)
    assert pytest.approx(np.linalg.norm(a2[:3])) == 1.0
    assert a2[3] > 0
    assert np.pi > a2[3]


def test_check_compact_axis_angle():
    """Test input validation for compact axis-angle representation."""
    a_list = [0, 0, 0]
    a = libtrans.normalize_compact_axang(np.array(a_list))
    assert_array_almost_equal(a_list, a)

    rng = np.random.default_rng(0)
    a = libtrans.normalize_vector(pr.random_vector(rng, 3))
    a *= np.pi + rng.standard_normal() * 4.0 * np.pi
    a2 = libtrans.normalize_compact_axang(a)
    pr.assert_compact_axis_angle_equal(a, a2)
    assert np.pi > np.linalg.norm(a2) > 0


def test_check_quaternion():
    """Test input validation for quaternion representation."""
    q_list = np.array([1, 0, 0, 0])
    q = libtrans.normalize_quat(q_list)
    assert_array_almost_equal(q_list, q)

    rng = np.random.default_rng(0)
    q = rng.standard_normal(4)
    q = libtrans.normalize_quat(q)
    assert pytest.approx(np.linalg.norm(q)) == 1.0


def test_pick_closest_quaternion():
    rng = np.random.default_rng(483)
    for _ in range(10):
        q = pr.random_quaternion(rng)
        assert_array_almost_equal(libtrans.pick_closest_quaternion(q, q), q)
        assert_array_almost_equal(libtrans.pick_closest_quaternion(-q, q), q)


def test_quaternion_dist():
    """Test angular metric of quaternions."""
    rng = np.random.default_rng(0)

    for _ in range(5):
        q1 = libtrans.axang_to_quat(pr.random_axis_angle(rng))
        q2 = libtrans.axang_to_quat(pr.random_axis_angle(rng))
        q1_to_q1 = libtrans.quat_distance(q1, q1)
        assert pytest.approx(q1_to_q1) == 0.0
        q2_to_q2 = libtrans.quat_distance(q2, q2)
        assert pytest.approx(q2_to_q2) == 0.0
        q1_to_q2 = libtrans.quat_distance(q1, q2)
        q2_to_q1 = libtrans.quat_distance(q2, q1)
        assert pytest.approx(q1_to_q2) == q2_to_q1
        assert 2.0 * np.pi > q1_to_q2


def test_quaternion_dist_for_identical_rotations():
    """Test angular metric of quaternions q and -q."""
    rng = np.random.default_rng(0)

    for _ in range(5):
        q = libtrans.axang_to_quat(pr.random_axis_angle(rng))
        assert_array_almost_equal(libtrans.quat_to_rmat(q),
                                  libtrans.quat_to_rmat(-q))
        assert libtrans.quat_distance(q, -q) == 0.0


def test_quaternion_dist_for_almost_identical_rotations():
    """Test angular metric of quaternions q and ca. -q."""
    rng = np.random.default_rng(0)

    for _ in range(5):
        a = pr.random_axis_angle(rng)
        q1 = libtrans.axang_to_quat(a)
        r = 1e-4 * rng.standard_normal(4)
        q2 = -libtrans.axang_to_quat(a + r)
        assert pytest.approx(libtrans.quat_distance(q1, q2), abs=1e-3) == 0.0


def test_interpolate_quaternion():
    """Test interpolation between two quaternions with slerp."""
    n_steps = 10
    rng = np.random.default_rng(0)
    a1 = pr.random_axis_angle(rng)
    a2 = pr.random_axis_angle(rng)
    q1 = libtrans.axang_to_quat(a1)
    q2 = libtrans.axang_to_quat(a2)

    traj_q = [libtrans.quat_slerp(q1, q2, t)
              for t in np.linspace(0, 1, n_steps)]
    traj_R = [libtrans.quat_to_rmat(q) for q in traj_q]
    R_diff = np.diff(traj_R, axis=0)
    R_diff_norms = [np.linalg.norm(Rd) for Rd in R_diff]
    assert_array_almost_equal(R_diff_norms,
                              R_diff_norms[0] * np.ones(n_steps - 1))


def test_interpolate_quaternion_shortest_path():
    """Test SLERP between similar quternions with opposite sign."""
    n_steps = 10
    rng = np.random.default_rng(2323)
    q1 = pr.random_quaternion(rng)
    a1 = libtrans.quat_to_axang(q1)
    a2 = np.r_[a1[:3], a1[3] * 1.1]
    q2 = libtrans.axang_to_quat(a2)

    if np.sign(q1[0]) != np.sign(q2[0]):
        q2 *= -1.0
    traj_q = [libtrans.quat_slerp(q1, q2, t)
              for t in np.linspace(0, 1, n_steps)]
    path_length = np.sum([libtrans.quat_distance(r, s)
                          for r, s in zip(traj_q[:-1], traj_q[1:])])

    q2 *= -1.0
    traj_q_opposing = [libtrans.quat_slerp(q1, q2, t)
                       for t in np.linspace(0, 1, n_steps)]
    path_length_opposing = np.sum(
        [libtrans.quat_distance(r, s)
         for r, s in zip(traj_q_opposing[:-1], traj_q_opposing[1:])])

    assert path_length_opposing > path_length

    traj_q_opposing_corrected = [
        libtrans.quat_slerp(q1, q2, t, shortest_path=True)
        for t in np.linspace(0, 1, n_steps)]
    path_length_opposing_corrected = np.sum(
        [libtrans.quat_distance(r, s)
         for r, s in zip(traj_q_opposing_corrected[:-1],
                         traj_q_opposing_corrected[1:])])

    assert pytest.approx(path_length_opposing_corrected) == path_length


def test_interpolate_same_quaternion():
    """Test interpolation between the same quaternion rotation.

    See issue #45.
    """
    n_steps = 3
    rng = np.random.default_rng(42)
    a = pr.random_axis_angle(rng)
    q = libtrans.axang_to_quat(a)
    traj = [libtrans.quat_slerp(q, q, t) for t in np.linspace(0, 1, n_steps)]
    assert len(traj) == n_steps
    assert_array_almost_equal(traj[0], q)
    assert_array_almost_equal(traj[1], q)
    assert_array_almost_equal(traj[2], q)


def test_interpolate_shortest_path_same_quaternion():
    """Test interpolate along shortest path with same quaternion."""
    rng = np.random.default_rng(8353)
    q = pr.random_quaternion(rng)
    q_interpolated = libtrans.quat_slerp(q, q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)

    q = np.array([0.0, 1.0, 0.0, 0.0])
    q_interpolated = libtrans.quat_slerp(q, -q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)

    q = np.array([0.0, 0.0, 1.0, 0.0])
    q_interpolated = libtrans.quat_slerp(q, -q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)

    q = np.array([0.0, 0.0, 0.0, 1.0])
    q_interpolated = libtrans.quat_slerp(q, -q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)


def test_concatenate_quaternions():
    """Test concatenation of two quaternions."""
    # Until ea9adc5, this combination of a list and a numpy array raised
    # a ValueError:
    # NOTE: Because it is the C library wrapper,
    #       we always pass the numpy array.
    # q1 = [1, 0, 0, 0]
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([0, 0, 0, 1])
    q12 = libtrans.quat_product(q1, q2)
    assert_array_almost_equal(q12, np.array([0, 0, 0, 1]))

    rng = np.random.default_rng(0)
    for _ in range(5):
        q1 = libtrans.axang_to_quat(pr.random_axis_angle(rng))
        q2 = libtrans.axang_to_quat(pr.random_axis_angle(rng))

        R1 = libtrans.quat_to_rmat(q1)
        R2 = libtrans.quat_to_rmat(q2)

        q12 = libtrans.quat_product(q1, q2)
        R12 = np.dot(R1, R2)
        q12R = libtrans.rmat_to_quat(R12)

        pr.assert_quaternion_equal(q12, q12R)


def test_quaternion_hamilton():
    """Test if quaternion multiplication follows Hamilton's convention."""
    q_ij = libtrans.quat_product(pr.q_i, pr.q_j)
    assert_array_equal(pr.q_k, q_ij)
    q_ijk = libtrans.quat_product(q_ij, pr.q_k)
    assert_array_equal(-pr.q_id, q_ijk)


def test_quaternion_conjugate():
    """Test quaternion conjugate."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = pr.random_quaternion(rng)
        v = pr.random_vector(rng)
        vq = libtrans.quat_prod_vec(q, v)
        vq2 = libtrans.quat_product(libtrans.quat_product(
            q, np.hstack(([0], v))), libtrans.quat_conjugate(q))[1:]
        assert_array_almost_equal(vq, vq2)


def test_quaternion_invert():
    """Test unit quaternion inversion with conjugate."""
    q = np.array([0.58183503, -0.75119889, -0.24622332, 0.19116072])
    q_inv = libtrans.quat_conjugate(q)
    q_q_inv = libtrans.quat_product(q, q_inv)
    assert_array_almost_equal(pr.q_id, q_q_inv)


def test_quaternion_rotation_consistent_with_multiplication():
    """Test if quaternion rotation and multiplication are Hamiltonian."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = pr.random_vector(rng)
        q = pr.random_quaternion(rng)
        v_im = np.hstack(((0.0,), v))
        qv_mult = libtrans.quat_product(
            q, libtrans.quat_product(v_im, libtrans.quat_conjugate(q)))[1:]
        qv_rot = libtrans.quat_prod_vec(q, v)
        assert_array_almost_equal(qv_mult, qv_rot)