import numpy as np
import pytransform3d.rotations as pr
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from lib_wrapper import LibWrapper, AXIS2NUM


libtrans = LibWrapper()


def test_passive_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    R = libtrans.ang_to_rmat(0, -0.5 * np.pi, True)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    R = libtrans.ang_to_rmat(0, 0.5 * np.pi, True)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    R = libtrans.ang_to_rmat(1, -0.5 * np.pi, True)
    assert_array_almost_equal(R, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    R = libtrans.ang_to_rmat(1, 0.5 * np.pi, True)
    assert_array_almost_equal(R, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))

    R = libtrans.ang_to_rmat(2, -0.5 * np.pi, True)
    assert_array_almost_equal(R, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    R = libtrans.ang_to_rmat(2, 0.5 * np.pi, True)
    assert_array_almost_equal(R, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))


def test_active_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    rng = np.random.default_rng(21)
    for _ in range(20):
        basis = rng.integers(0, 3)
        angle = 2.0 * np.pi * rng.random() - np.pi
        R_passive = libtrans.ang_to_rmat(basis, angle, True)
        R_active = libtrans.ang_to_rmat(basis, angle, False)
        assert_array_almost_equal(R_active, R_passive.T)


def test_active_matrix_from_intrinsic_euler_zxz():
    """Test conversion from intrinsic zxz Euler angles."""
    i, j, k = AXIS2NUM['Z'], AXIS2NUM['X'], AXIS2NUM['Z']
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0.5 * np.pi]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    )


def test_active_matrix_from_extrinsic_euler_zxz():
    """Test conversion from extrinsic zxz Euler angles."""
    i, j, k = AXIS2NUM['Z'], AXIS2NUM['X'], AXIS2NUM['Z']
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0.5 * np.pi]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    )


def test_active_matrix_from_intrinsic_euler_zyz():
    """Test conversion from intrinsic zyz Euler angles."""
    i, j, k = AXIS2NUM['Z'], AXIS2NUM['Y'], AXIS2NUM['Z']
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0]),
            i,
            j,
            k,
            False
        ),
        libtrans.euler_to_rmat(np.array([0.5 * np.pi, 0, 0]), 2, 1, 2, False)
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0.5 * np.pi]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
            i,
            j,
            k,
            False
        ),
        np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
    )


def test_active_matrix_from_extrinsic_euler_zyz():
    """Test conversion from extrinsic zyz Euler angles."""
    i, j, k = AXIS2NUM['Z'], AXIS2NUM['Y'], AXIS2NUM['Z']
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )

    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0.5 * np.pi]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
    )


def test_active_matrix_from_extrinsic_roll_pitch_yaw():
    """Test conversion from roll, pitch, yaw."""
    i, j, k = AXIS2NUM['X'], AXIS2NUM['Y'], AXIS2NUM['Z']
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0]),
            i,
            j,
            k,
            True
        ),
        libtrans.euler_to_rmat(np.array([0.5 * np.pi, 0, 0]), 0, 1, 2, True)
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0, 0.5 * np.pi]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        libtrans.euler_to_rmat(
            np.array([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
            i,
            j,
            k,
            True
        ),
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    )


def test_active_matrix_from_intrinsic_zyx():
    """Test conversion from intrinsic zyx Euler angles."""
    i, j, k = AXIS2NUM['Z'], AXIS2NUM['Y'], AXIS2NUM['X']
    rng = np.random.default_rng(844)
    for _ in range(5):
        euler_zyx = ((rng.random(3) - 0.5) *
                     np.array([np.pi, 0.5 * np.pi, np.pi]))
        s = np.sin(euler_zyx)
        c = np.cos(euler_zyx)
        R_from_formula = np.array([
            [c[0] * c[1], c[0] * s[1] * s[2] - s[0] * c[2],
             c[0] * s[1] * c[2] + s[0] * s[2]],
            [s[0] * c[1], s[0] * s[1] * s[2] + c[0] * c[2],
             s[0] * s[1] * c[2] - c[0] * s[2]],
            [-s[1], c[1] * s[2], c[1] * c[2]]
        ])  # See Lynch, Park: Modern Robotics, page 576

        # Normal case, we can reconstruct original angles
        R = libtrans.euler_to_rmat(euler_zyx, i, j, k, False)
        assert_array_almost_equal(R_from_formula, R)
        euler_zyx2 = libtrans.rmat_to_euler(R, i, j, k, False)
        assert_array_almost_equal(euler_zyx, euler_zyx2)

        # Gimbal lock 1, infinite solutions with constraint
        # alpha - gamma = constant
        euler_zyx[1] = 0.5 * np.pi
        R = libtrans.euler_to_rmat(euler_zyx, i, j, k, False)
        euler_zyx2 = libtrans.rmat_to_euler(R, i, j, k, False)
        assert pytest.approx(euler_zyx2[1]) == 0.5 * np.pi
        assert (pytest.approx(euler_zyx[0] - euler_zyx[2])
                == euler_zyx2[0] - euler_zyx2[2])

        # Gimbal lock 2, infinite solutions with constraint
        # alpha + gamma = constant
        euler_zyx[1] = -0.5 * np.pi
        R = libtrans.euler_to_rmat(euler_zyx, i, j, k, False)
        euler_zyx2 = libtrans.rmat_to_euler(R, i, j, k, False)
        assert pytest.approx(euler_zyx2[1]) == -0.5 * np.pi
        assert (pytest.approx(euler_zyx[0] + euler_zyx[2])
                == euler_zyx2[0] + euler_zyx2[2])


def test_active_matrix_from_extrinsic_zyx():
    """Test conversion from extrinsic zyx Euler angles."""
    i, j, k = AXIS2NUM['Z'], AXIS2NUM['Y'], AXIS2NUM['X']
    rng = np.random.default_rng(844)
    for _ in range(5):
        euler_zyx = ((rng.random(3) - 0.5)
                     * np.array([np.pi, 0.5 * np.pi, np.pi]))

        # Normal case, we can reconstruct original angles
        R = libtrans.euler_to_rmat(euler_zyx, i, j, k, True)
        euler_zyx2 = libtrans.rmat_to_euler(R, i, j, k, True)
        assert_array_almost_equal(euler_zyx, euler_zyx2)
        R2 = libtrans.euler_to_rmat(euler_zyx2, i, j, k, True)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 1, infinite solutions with constraint
        # alpha + gamma = constant
        euler_zyx[1] = 0.5 * np.pi
        R = libtrans.euler_to_rmat(euler_zyx, i, j, k, True)
        euler_zyx2 = libtrans.rmat_to_euler(R, i, j, k, True)
        assert pytest.approx(euler_zyx2[1]) == 0.5 * np.pi
        assert pytest.approx(
            euler_zyx[0] + euler_zyx[2]) == euler_zyx2[0] + euler_zyx2[2]
        R2 = libtrans.euler_to_rmat(euler_zyx2, i, j, k, True)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 2, infinite solutions with constraint
        # alpha - gamma = constant
        euler_zyx[1] = -0.5 * np.pi
        R = libtrans.euler_to_rmat(euler_zyx, i, j, k, True)
        euler_zyx2 = libtrans.rmat_to_euler(R, i, j, k, True)
        assert pytest.approx(euler_zyx2[1]) == -0.5 * np.pi
        assert pytest.approx(
            euler_zyx[0] - euler_zyx[2]) == euler_zyx2[0] - euler_zyx2[2]
        R2 = libtrans.euler_to_rmat(euler_zyx2, i, j, k, True)
        assert_array_almost_equal(R, R2)


def test_all_euler_matrix_conversions():
    euler_axes = [
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 2, 1],
        [2, 1, 2],
        [2, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1]
    ]
    for ea in euler_axes:
        proper_euler = (ea[0] == ea[2])
        for extrinsic in [False, True]:
            rng = np.random.default_rng(844)
            for _ in range(5):
                euler = ((rng.random(3) - 0.5)
                        * np.array([np.pi, 0.5 * np.pi, np.pi]))
                if proper_euler:
                    euler[1] += 0.5 * np.pi

                # Normal case, we can reconstruct original angles
                R = libtrans.euler_to_rmat(
                    euler,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                euler2 = libtrans.rmat_to_euler(
                    R,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                assert_array_almost_equal(euler, euler2)
                R2 = libtrans.euler_to_rmat(
                    euler2,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                assert_array_almost_equal(R, R2)

                # Gimbal lock 1
                if proper_euler:
                    euler[1] = np.pi
                else:
                    euler[1] = 0.5 * np.pi
                R = libtrans.euler_to_rmat(
                    euler,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                euler2 = libtrans.rmat_to_euler(
                    R,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                assert pytest.approx(euler[1]) == euler2[1]
                R2 = libtrans.euler_to_rmat(
                    euler2,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                assert_array_almost_equal(R, R2)

                # Gimbal lock 2
                if proper_euler:
                    euler[1] = 0.0
                else:
                    euler[1] = -0.5 * np.pi
                R = libtrans.euler_to_rmat(
                    euler,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                euler2 = libtrans.rmat_to_euler(
                    R,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                assert pytest.approx(euler[1]) == euler2[1]
                R2 = libtrans.euler_to_rmat(
                    euler2,
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                assert_array_almost_equal(R, R2)


def test_from_quaternion():
    """Test conversion from quaternion to Euler angles."""
    rng = np.random.default_rng(32)

    euler_axes = [
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 2, 1],
        [2, 1, 2],
        [2, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1]
    ]

    for ea in euler_axes:
        for extrinsic in [False, True]:
            for _ in range(5):
                e = rng.random(3)
                e[0] = 2.0 * np.pi * e[0] - np.pi
                e[1] = np.pi * e[1]
                e[2] = 2.0 * np.pi * e[2] - np.pi

                proper_euler = ea[0] == ea[2]
                if proper_euler:
                    e[1] -= np.pi / 2.0

                # normal case
                q = libtrans.rmat_to_quat(
                    libtrans.euler_to_rmat(
                        e,
                        ea[0],
                        ea[1],
                        ea[2],
                        extrinsic
                    )
                )

                e1 = libtrans.quat_to_euler(q, ea[0], ea[1], ea[2], extrinsic)
                e2 = libtrans.rmat_to_euler(
                    libtrans.quat_to_rmat(q),
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                assert_array_almost_equal(
                    e1, e2, err_msg=f"axes: {ea}, extrinsic: {extrinsic}")

                # first singularity
                e[1] = 0.0
                q = libtrans.rmat_to_quat(
                    libtrans.euler_to_rmat(e, ea[0], ea[1], ea[2], extrinsic)
                )

                R1 = libtrans.euler_to_rmat(
                    libtrans.quat_to_euler(q, ea[0], ea[1], ea[2], extrinsic),
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                R2 = libtrans.quat_to_rmat(q)
                assert_array_almost_equal(
                    R1, R2, err_msg=f"axes: {ea}, extrinsic: {extrinsic}")
                
                # second singularity
                e[1] = np.pi
                q = libtrans.rmat_to_quat(
                    libtrans.euler_to_rmat(e, ea[0], ea[1], ea[2], extrinsic)
                )

                R1 = libtrans.euler_to_rmat(
                    libtrans.quat_to_euler(q, ea[0], ea[1], ea[2], extrinsic),
                    ea[0],
                    ea[1],
                    ea[2],
                    extrinsic
                )
                R2 = libtrans.quat_to_rmat(q)
                assert_array_almost_equal(
                    R1, R2, err_msg=f"axes: {ea}, extrinsic: {extrinsic}")


def test_conversions_matrix_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    i, j, k = AXIS2NUM['X'], AXIS2NUM['Y'], AXIS2NUM['Z']
    R = np.eye(3)
    a = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    R = libtrans.euler_to_rmat(
        np.array([-np.pi, -np.pi, 0.0]),
        i,
        j,
        k,
        False
    )
    a = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    R = libtrans.euler_to_rmat(
        np.array([-np.pi, 0.0, -np.pi]),
        i,
        j,
        k,
        False
    )
    a = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, np.array([0, 1, 0, np.pi]))

    R = libtrans.euler_to_rmat(
        np.array([0.0, -np.pi, -np.pi]),
        i,
        j,
        k,
        False
    )
    a = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, np.pi]))

    a = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, np.pi])
    R = libtrans.axang_to_rmat(a)
    a2 = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a2, a)

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = pr.random_axis_angle(rng)
        R = libtrans.axang_to_rmat(a)
        assert libtrans.is_rmat(R)

        a2 = libtrans.rmat_to_axang(R)
        pr.assert_axis_angle_equal(a, a2)

        R2 = libtrans.axang_to_rmat(a2)
        assert_array_almost_equal(R, R2)
        assert libtrans.is_rmat(R2)


def test_compare_axis_angle_from_matrix_to_lynch_park():
    """Compare log(R) to the version of Lynch, Park: Modern Robotics."""
    R = np.eye(3)
    a = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, [0, 0, 0, 0])

    R = libtrans.ang_to_rmat(2, np.pi, True)
    assert pytest.approx(np.trace(R)) == -1
    a = libtrans.rmat_to_axang(R)
    axis = (1.0 / np.sqrt(2.0 * (1 + R[2, 2]))
            * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]))
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    R = libtrans.ang_to_rmat(1, np.pi, True)
    assert pytest.approx(np.trace(R)) == -1
    a = libtrans.rmat_to_axang(R)
    axis = (1.0 / np.sqrt(2.0 * (1 + R[1, 1]))
            * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]]))
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    R = libtrans.ang_to_rmat(0, np.pi, True)
    assert pytest.approx(np.trace(R)) == -1
    a = libtrans.rmat_to_axang(R)
    axis = (1.0 / np.sqrt(2.0 * (1 + R[0, 0]))
            * np.array([1 + R[0, 0], R[1, 0], R[2, 0]]))
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    # normal case is omitted here


def test_conversions_matrix_compact_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    i, j, k = AXIS2NUM['X'], AXIS2NUM['Y'], AXIS2NUM['Z']
    R = np.eye(3)
    a = libtrans.rmat_to_compact_axang(R)
    pr.assert_compact_axis_angle_equal(a, np.zeros(3))

    R = libtrans.euler_to_rmat(
        np.array([-np.pi, -np.pi, 0.0]),
        i,
        j,
        k,
        False
    )
    a = libtrans.rmat_to_compact_axang(R)
    pr.assert_compact_axis_angle_equal(a, np.array([0, 0, np.pi]))

    R = libtrans.euler_to_rmat(
        np.array([-np.pi, 0.0, -np.pi]),
        i,
        j,
        k,
        False
    )
    a = libtrans.rmat_to_compact_axang(R)
    pr.assert_compact_axis_angle_equal(a, np.array([0, np.pi, 0]))

    R = libtrans.euler_to_rmat(
        np.array([0.0, -np.pi, -np.pi]),
        i,
        j,
        k,
        False
    )
    a = libtrans.rmat_to_compact_axang(R)
    pr.assert_compact_axis_angle_equal(a, np.array([np.pi, 0, 0]))

    a = np.array([np.sqrt(0.5) * np.pi, np.sqrt(0.5) * np.pi, 0.0])
    R = libtrans.compact_axang_to_rmat(a)
    a2 = libtrans.rmat_to_compact_axang(R)
    pr.assert_compact_axis_angle_equal(a2, a)

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = pr.random_compact_axis_angle(rng)
        R = libtrans.compact_axang_to_rmat(a)
        assert libtrans.is_rmat(R)

        a2 = libtrans.rmat_to_compact_axang(R)
        pr.assert_compact_axis_angle_equal(a, a2)

        R2 = libtrans.compact_axang_to_rmat(a2)
        assert_array_almost_equal(R, R2)
        assert libtrans.is_rmat(R2)


def test_active_rotation_is_default():
    """Test that rotations are active by default."""
    Rx = libtrans.ang_to_rmat(0, 0.5 * np.pi, False)
    ax = np.array([1, 0, 0, 0.5 * np.pi])
    qx = libtrans.axang_to_quat(ax)
    assert_array_almost_equal(Rx, libtrans.axang_to_rmat(ax))
    assert_array_almost_equal(Rx, libtrans.quat_to_rmat(qx))

    Ry = libtrans.ang_to_rmat(1, 0.5 * np.pi, False)
    ay = np.array([0, 1, 0, 0.5 * np.pi])
    qy = libtrans.axang_to_quat(ay)
    assert_array_almost_equal(Ry, libtrans.axang_to_rmat(ay))
    assert_array_almost_equal(Ry, libtrans.quat_to_rmat(qy))

    Rz = libtrans.ang_to_rmat(2, 0.5 * np.pi, False)
    az = np.array([0, 0, 1, 0.5 * np.pi])
    qz = libtrans.axang_to_quat(az)
    assert_array_almost_equal(Rz, libtrans.axang_to_rmat(az))
    assert_array_almost_equal(Rz, libtrans.quat_to_rmat(qz))


def test_issue43():
    """Test axis_angle_from_matrix() with angles close to 0 and pi."""
    a = np.array([-1., 1., 1., np.pi - 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = libtrans.axang_to_rmat(a)
    a2 = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = libtrans.axang_to_rmat(a)
    a2 = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., np.pi + 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = libtrans.axang_to_rmat(a)
    a2 = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., -5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = libtrans.axang_to_rmat(a)
    a2 = libtrans.rmat_to_axang(R)
    pr.assert_axis_angle_equal(a, a2)


def test_issue43_numerical_precision():
    """Test numerical precision of angles close to 0 and pi."""
    a = np.array([1., 1., 1., np.pi - 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = libtrans.axang_to_rmat(a)
    a2 = libtrans.rmat_to_axang(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert axis_dist < 1e-10
    assert abs(a[3] - a2[3]) < 1e-8

    a = np.array([1., 1., 1., 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = libtrans.axang_to_rmat(a)
    a2 = libtrans.rmat_to_axang(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert axis_dist < 1e-10
    assert abs(a[3] - a2[3]) < 1e-8


def test_conversions_matrix_axis_angle_continuous():
    """Test continuous conversions between rotation matrix and axis-angle."""
    for angle in np.arange(3.1, 3.2, 0.01):
        a = np.array([1.0, 0.0, 0.0, angle])
        R = libtrans.axang_to_rmat(a)
        assert libtrans.is_rmat(R)

        a2 = libtrans.rmat_to_axang(R)
        pr.assert_axis_angle_equal(a, a2)

        R2 = libtrans.axang_to_rmat(a2)
        assert_array_almost_equal(R, R2)
        assert libtrans.is_rmat(R2)


def test_conversions_matrix_quaternion():
    """Test conversions between rotation matrix and quaternion."""
    R = np.eye(3)
    a = libtrans.rmat_to_axang(R)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))

    rng = np.random.default_rng(0)
    for _ in range(5):
        q = pr.random_quaternion(rng)
        R = libtrans.quat_to_rmat(q)
        assert libtrans.is_rmat(R)

        q2 = libtrans.rmat_to_quat(R)
        pr.assert_quaternion_equal(q, q2)

        R2 = libtrans.quat_to_rmat(q2)
        assert_array_almost_equal(R, R2)
        assert libtrans.is_rmat(R2)


def test_matrix_from_quaternion_hamilton():
    """Test if the conversion from quaternion to matrix is Hamiltonian."""
    q = np.sqrt(0.5) * np.array([1, 0, 0, 1])
    R = libtrans.quat_to_rmat(q)
    assert_array_almost_equal(
        np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]),
        R
    )


def test_quaternion_from_matrix_180():
    """Test for bug in conversion from 180 degree rotations."""
    a = np.array([1.0, 0.0, 0.0, np.pi])
    q = libtrans.axang_to_quat(a)
    R = libtrans.axang_to_rmat(a)
    q_from_R = libtrans.rmat_to_quat(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    q = libtrans.axang_to_quat(a)
    R = libtrans.axang_to_rmat(a)
    q_from_R = libtrans.rmat_to_quat(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    q = libtrans.axang_to_quat(a)
    R = libtrans.axang_to_rmat(a)
    q_from_R = libtrans.rmat_to_quat(R)
    assert_array_almost_equal(q, q_from_R)


def test_quaternion_from_matrix_180_not_axis_aligned():
    """Test for bug in rotation by 180 degrees around arbitrary axes."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = pr.random_axis_angle(rng)
        a[3] = np.pi
        q = libtrans.axang_to_quat(a)
        R = libtrans.axang_to_rmat(a)
        q_from_R = libtrans.rmat_to_quat(R)
        pr.assert_quaternion_equal(q, q_from_R)


def test_quaternion_from_extrinsic_euler_xyz():
    """Test quaternion_from_extrinsic_euler_xyz."""
    i, j, k = AXIS2NUM['X'], AXIS2NUM['Y'], AXIS2NUM['Z']
    rng = np.random.default_rng(0)
    for _ in range(10):
        e = rng.uniform(-100, 100, [3])
        q = libtrans.euler_to_quat(e, i, j, k, True)
        R_from_q = libtrans.quat_to_rmat(q)
        R_from_e = libtrans.euler_to_rmat(e, i, j, k, True)
        assert_array_almost_equal(R_from_q, R_from_e)


def test_conversions_axis_angle_quaternion():
    """Test conversions between axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = libtrans.quat_to_axang(q)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))
    q2 = libtrans.axang_to_quat(a)
    assert_array_almost_equal(q2, q)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        q = libtrans.axang_to_quat(a)
        a2 = libtrans.quat_to_axang(q)
        assert_array_almost_equal(a, a2)

        q2 = libtrans.axang_to_quat(a2)
        pr.assert_quaternion_equal(q, q2)


def test_axis_angle_from_compact_axis_angle():
    """Test conversion from compact axis-angle representation."""
    ca = [0.0, 0.0, 0.0]
    a = libtrans.compact_axang_to_axang(np.array(ca))
    assert_array_almost_equal(a, np.array([1.0, 0.0, 0.0, 0.0]))

    rng = np.random.default_rng(1)
    for _ in range(5):
        ca = pr.random_compact_axis_angle(rng)
        a = libtrans.compact_axang_to_axang(ca)
        assert pytest.approx(np.linalg.norm(ca)) == a[3]
        assert_array_almost_equal(ca[:3] / np.linalg.norm(ca), a[:3])


def test_compact_axis_angle():
    """Test conversion to compact axis-angle representation."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    ca = libtrans.axang_to_compact_axang(a)
    assert_array_almost_equal(ca, np.zeros(3))

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        ca = libtrans.axang_to_compact_axang(a)
        assert_array_almost_equal(libtrans.normalize_vector(ca), a[:3])
        assert pytest.approx(np.linalg.norm(ca)) == a[3]


def test_conversions_compact_axis_angle_quaternion():
    """Test conversions between compact axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = libtrans.quat_to_compact_axang(q)
    assert_array_almost_equal(a, np.array([0, 0, 0]))
    q2 = libtrans.compact_axang_to_quat(a)
    assert_array_almost_equal(q2, q)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_compact_axis_angle(rng)
        q = libtrans.compact_axang_to_quat(a)

        a2 = libtrans.quat_to_compact_axang(q)
        assert_array_almost_equal(a, a2)

        q2 = libtrans.compact_axang_to_quat(a2)
        pr.assert_quaternion_equal(q, q2)


def test_quaternion_conventions():
    """Test conversion of quaternion between wxyz and xyzw."""
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    q_xyzw = libtrans.quat_wxyz_to_xyzw(q_wxyz)
    assert_array_equal(q_xyzw, np.array([0.0, 0.0, 0.0, 1.0]))
    q_wxyz2 = libtrans.quat_xyzw_to_wxyz(q_xyzw)
    assert_array_equal(q_wxyz, q_wxyz2)

    rng = np.random.default_rng(42)
    q_wxyz_random = pr.random_quaternion(rng)
    q_xyzw_random = libtrans.quat_wxyz_to_xyzw(q_wxyz_random)
    assert_array_equal(q_xyzw_random[:3], q_wxyz_random[1:])
    assert q_xyzw_random[3] == q_wxyz_random[0]
    q_wxyz_random2 = libtrans.quat_xyzw_to_wxyz(q_xyzw_random)
    assert_array_equal(q_wxyz_random, q_wxyz_random2)


def test_concatenate_quaternions():
    """Test concatenation of two quaternions."""
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
        R12 = libtrans.mat_multiply(R1, R2)
        q12R = libtrans.rmat_to_quat(R12)

        pr.assert_quaternion_equal(q12, q12R)


def test_quaternion_hamilton():
    """Test if quaternion multiplication follows Hamilton's convention."""
    q_ij = libtrans.quat_product(pr.q_i, pr.q_j)
    assert_array_equal(pr.q_k, q_ij)
    q_ijk = libtrans.quat_product(q_ij, pr.q_k)
    assert_array_equal(-pr.q_id, q_ijk)


def test_quaternion_rotation():
    """Test quaternion rotation."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        q = libtrans.axang_to_quat(a)
        R = libtrans.quat_to_rmat(q)
        v = pr.random_vector(rng)
        vR = np.dot(R, v)
        vq = libtrans.quat_prod_vec(q, v)
        assert_array_almost_equal(vR, vq)


def test_quaternion_conjugate():
    """Test quaternion conjugate."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = pr.random_quaternion(rng)
        v = pr.random_vector(rng)
        vq = libtrans.quat_prod_vec(q, v)
        vq2 = libtrans.quat_product(
            libtrans.quat_product(q, np.hstack(([0], v))),
            libtrans.quat_conjugate(q)
        )[1:]
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


def test_id_rot():
    """Test equivalence of constants that represent no rotation."""
    assert_array_almost_equal(pr.R_id, libtrans.axang_to_rmat(pr.a_id))
    assert_array_almost_equal(pr.R_id, libtrans.quat_to_rmat(pr.q_id))


def test_check_matrix_threshold():
    """Test matrix threshold.

    See issue #54.
    """
    R = np.array([
        [-9.15361835e-01, 4.01808328e-01, 2.57475872e-02],
        [5.15480570e-02, 1.80374088e-01, -9.82246499e-01],
        [-3.99318925e-01, -8.97783496e-01, -1.85819250e-01]])
    assert libtrans.is_rmat(R)


def test_axis_angle_from_matrix_cos_angle_greater_1():
    R = np.array([
        [1.0000000000000004, -1.4402617650886727e-08, 2.3816502339526408e-08],
        [1.4402617501592725e-08, 1.0000000000000004, 1.2457848566326355e-08],
        [-2.3816502529500374e-08, -1.2457848247850049e-08, 0.9999999999999999]
    ])
    a = libtrans.rmat_to_axang(R)
    assert not any(np.isnan(a))


def test_bug_189():
    """Test bug #189"""
    R = np.array([
        [-1.0000000000000004e+00, 2.8285718503485576e-16,
         1.0966597378775709e-16],
        [1.0966597378775709e-16, -2.2204460492503131e-16,
         1.0000000000000002e+00],
        [2.8285718503485576e-16, 1.0000000000000002e+00,
         -2.2204460492503131e-16]
    ])
    a1 = libtrans.rmat_to_compact_axang(R)
    a2 = libtrans.rmat_to_compact_axang(pr.norm_matrix(R))
    assert_array_almost_equal(a1, a2)


def test_bug_198():
    """Test bug #198"""
    R = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, -1]], dtype=float)
    a = libtrans.rmat_to_compact_axang(R)
    R2 = libtrans.compact_axang_to_rmat(a)
    assert_array_almost_equal(R, R2)


def test_quaternion_from_angle():
    """Quaternion from rotation around basis vectors."""
    rng = np.random.default_rng(22)
    for _ in range(20):
        basis = rng.integers(0, 3)
        angle = 2.0 * np.pi * rng.random() - np.pi
        R = libtrans.ang_to_rmat(basis, angle, False)
        q = libtrans.ang_to_quat(basis, angle)
        Rq = libtrans.quat_to_rmat(q)
        assert_array_almost_equal(R, Rq)


def test_quaternion_from_euler():
    """Quaternion from Euler angles."""
    euler_axes = [
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 2, 1],
        [2, 1, 2],
        [2, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1]
    ]
    rng = np.random.default_rng(83)
    for ea in euler_axes:
        for extrinsic in [False, True]:
            for _ in range(5):
                e = rng.random(3)
                e[0] = 2.0 * np.pi * e[0] - np.pi
                e[1] = np.pi * e[1]
                e[2] = 2.0 * np.pi * e[2] - np.pi

                proper_euler = ea[0] == ea[2]
                if proper_euler:
                    e[1] -= np.pi / 2.0

                q = libtrans.euler_to_quat(
                    e, ea[0], ea[1], ea[2], extrinsic)
                e2 = libtrans.quat_to_euler(
                    q, ea[0], ea[1], ea[2], extrinsic)
                q2 = libtrans.euler_to_quat(
                    e2, ea[0], ea[1], ea[2], extrinsic)

                pr.assert_quaternion_equal(q, q2)


def test_general_matrix_euler_conversions():
    """General conversion algorithms between matrix and Euler angles."""
    rng = np.random.default_rng(22)

    euler_axes = [
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 2, 1],
        [2, 1, 2],
        [2, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1]
    ]
    for ea in euler_axes:
        for extrinsic in [False, True]:
            for _ in range(5):
                e = rng.random(3)
                e[0] = 2.0 * np.pi * e[0] - np.pi
                e[1] = np.pi * e[1]
                e[2] = 2.0 * np.pi * e[2] - np.pi

                proper_euler = ea[0] == ea[2]
                if proper_euler:
                    e[1] -= np.pi / 2.0

                q = libtrans.euler_to_quat(
                    e, ea[0], ea[1], ea[2], extrinsic)
                R = libtrans.euler_to_rmat(e, ea[0], ea[1], ea[2], extrinsic)
                q_R = libtrans.rmat_to_quat(R)
                pr.assert_quaternion_equal(
                    q, q_R, err_msg=f"axes: {ea}, extrinsic: {extrinsic}")

                e_R = libtrans.rmat_to_euler(R, ea[0], ea[1], ea[2], extrinsic)
                e_q = libtrans.quat_to_euler(
                    q, ea[0], ea[1], ea[2], extrinsic)

                R_R = libtrans.euler_to_rmat(
                    e_R, ea[0], ea[1], ea[2], extrinsic)
                R_q = libtrans.euler_to_rmat(
                    e_q, ea[0], ea[1], ea[2], extrinsic)
                assert_array_almost_equal(R_R, R_q)


def test_euler_from_quaternion_edge_case():
    quaternion = np.array([0.57114154, -0.41689009, -0.57114154, -0.41689009])
    matrix = libtrans.quat_to_rmat(quaternion)
    euler_xyz = libtrans.rmat_to_euler(matrix, 0, 1, 2, True)
    assert not np.isnan(euler_xyz).all()
