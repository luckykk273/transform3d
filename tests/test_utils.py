import numpy as np
import pytransform3d.rotations as pr
from numpy.testing import assert_array_almost_equal
import pytest
from lib_wrapper import LibWrapper

libtrans = LibWrapper()


def test_norm_vector():
    """Test normalization of vectors."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = pr.random_vector(rng, 3)
        u = libtrans.normalize_vec(v)
        assert pytest.approx(np.linalg.norm(u)) == 1


def test_norm_zero_vector():
    """Test normalization of zero vector."""
    normalized = libtrans.normalize_vec(np.zeros(3))
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
        a_on_b = libtrans.vec_proj(a, np.zeros(3))
        assert_array_almost_equal(a_on_b, np.zeros(3))


def test_vector_projection():
    """Test orthogonal projection of one vector to another vector."""
    a = np.ones(3)
    a_on_unitx = libtrans.vec_proj(a, pr.unitx)
    assert_array_almost_equal(a_on_unitx, pr.unitx)
    assert pytest.approx(
        pr.angle_between_vectors(a_on_unitx, pr.unitx)) == 0.0

    a2_on_unitx = libtrans.vec_proj(2 * a, pr.unitx)
    assert_array_almost_equal(a2_on_unitx, 2 * pr.unitx)
    assert pytest.approx(
        pr.angle_between_vectors(a2_on_unitx, pr.unitx)) == 0.0
    
    a_on_unity = libtrans.vec_proj(a, pr.unity)
    assert_array_almost_equal(a_on_unity, pr.unity)
    assert pytest.approx(
        pr.angle_between_vectors(a_on_unity, pr.unity)) == 0.0

    minus_a_on_unity = libtrans.vec_proj(-a, pr.unity)
    assert_array_almost_equal(minus_a_on_unity, -pr.unity)
    assert pytest.approx(
        pr.angle_between_vectors(minus_a_on_unity, pr.unity)) == np.pi

    a_on_unitz = libtrans.vec_proj(a, pr.unitz)
    assert_array_almost_equal(a_on_unitz, pr.unitz)
    assert pytest.approx(
        pr.angle_between_vectors(a_on_unitz, pr.unitz)) == 0.0

    unitz_on_a = libtrans.vec_proj(pr.unitz, a)
    assert_array_almost_equal(unitz_on_a, np.ones(3) / 3.0)
    assert pytest.approx(pr.angle_between_vectors(unitz_on_a, a)) == 0.0

    unitx_on_unitx = libtrans.vec_proj(pr.unitx, pr.unitx)
    assert_array_almost_equal(unitx_on_unitx, pr.unitx)
    assert pytest.approx(
        pr.angle_between_vectors(unitx_on_unitx, pr.unitx)) == 0.0


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
    a = libtrans.normalize_vec(pr.random_vector(rng, 3))
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
