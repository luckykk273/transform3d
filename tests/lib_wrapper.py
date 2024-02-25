'''
lib_wrapper.py
The C library wrapper is defined in this file.
'''
import ctypes
import numpy as np


AXIS2NUM = {
    'X': 0,
    'Y': 1,
    'Z': 2
}


class Helper:
    '''
    This Helper class is not used because we update all (*)[N] poiner args to * pointer args.
    But still keep here maybe one day it will be helpful.
    '''
    def __init__(self) -> None:
        # necessary for clib
        self.c_double_ptr = ctypes.POINTER(ctypes.c_double)  # double *p
        self.c_double_3_array = ctypes.c_double * 3  # double arr[3]
        self.c_array_type = self.c_double_3_array * 3
        self.c_2d_array_ptr = ctypes.POINTER(self.c_array_type)  # double (*arr)[3]
    
    # transformation between C and Python
    def _get_c_2d_array(self):
        c_2d_array = self.c_array_type()  # double arr[3][3]
        return c_2d_array

    def np_2d_array_to_c_2d_array(self, np_2d_array: np.ndarray):
        np_2d_array = np_2d_array.astype(np.double)
        c_2d_array = self._get_c_2d_array()
        for i, row in enumerate(np_2d_array):
            c_2d_array[i] = self.c_double_3_array(*row)
        return c_2d_array

    def c_2d_array_to_np_2d_array(self, c_2d_array: ((ctypes.c_double * 3)*3)()):
        c_2d_list = [list(c_2d_array[i]) for i in range(3)]
        np_2d_array = np.array(c_2d_list, dtype=np.double)
        return np_2d_array


class LibWrapper:
    '''
    All C library is wrapped in this class.
    '''
    def __init__(self, libpath: str = '../build/dll/libtransform3d_dynamic.dll') -> None:
        self.libtrans = ctypes.cdll.LoadLibrary(libpath)
        
        # define arg types and res type
        # matrix.h
        self.libtrans.mat_multiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *m12
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m1
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m2
            ctypes.c_size_t, # const size_t r1
            ctypes.c_size_t, # const size_t c1
            ctypes.c_size_t, # const size_t c2
        ]
        self.libtrans.mat_multiply.restype = None

        self.libtrans.mat_transpose.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *mt
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m
            ctypes.c_size_t, # const size_t r
            ctypes.c_size_t, # const size_t c
        ]
        self.libtrans.mat_transpose.restype = None

        # vector.h
        self.libtrans.orthogonal_project.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *proj
            np.ctypeslib.ndpointer(dtype=np.double), # const double *u
            np.ctypeslib.ndpointer(dtype=np.double), # const double *v
            ctypes.c_size_t, # const size_t n
        ]
        self.libtrans.orthogonal_project.restype = None

        self.libtrans.normalize_vector.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *v_norm
            np.ctypeslib.ndpointer(dtype=np.double), # const double *v
            ctypes.c_size_t, # const size_t n
        ]
        self.libtrans.normalize_vector.restype = None

        self.libtrans.angle_between_vectors.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # const double *u
            np.ctypeslib.ndpointer(dtype=np.double), # const double *v
            ctypes.c_bool, # const bool fast
            ctypes.c_size_t, # const size_t n
        ]
        self.libtrans.angle_between_vectors.restype = ctypes.c_double

        # trans_utils.h
        # Helper functions
        self.libtrans.quat_prod_vec.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *qv
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
            np.ctypeslib.ndpointer(dtype=np.double), # const double *v
        ]
        self.libtrans.quat_prod_vec.restype = None

        self.libtrans.quat_product.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *pq
            np.ctypeslib.ndpointer(dtype=np.double), # double *p
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
        ]
        self.libtrans.quat_product.restype = None

        self.libtrans.quat_conjugate.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q_conj
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
        ]
        self.libtrans.quat_conjugate.restype = None

        self.libtrans.pick_closest_quaternion.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *closest
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q_target
        ]
        self.libtrans.pick_closest_quaternion.restype = None

        self.libtrans.quat_slerp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *slerp
            np.ctypeslib.ndpointer(dtype=np.double), # const double *start
            np.ctypeslib.ndpointer(dtype=np.double), # const double *end
            ctypes.c_double, # const double t
            ctypes.c_bool, # const bool shortest_path
        ]
        self.libtrans.quat_slerp.restype = None

        self.libtrans.quat_distance.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # const double *p
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
        ]
        self.libtrans.quat_distance.restype = ctypes.c_double

        # Validation functions
        self.libtrans.is_rmat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m
        ]
        self.libtrans.is_rmat.restype = ctypes.c_bool

        self.libtrans.normalize_ang.argtypes = [
            ctypes.c_double, # const double ang
        ]
        self.libtrans.normalize_ang.restype = ctypes.c_double

        self.libtrans.normalize_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a_norm
            np.ctypeslib.ndpointer(dtype=np.double), # const double *a
        ]
        self.libtrans.normalize_axang.restype = None

        self.libtrans.normalize_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca_norm
            np.ctypeslib.ndpointer(dtype=np.double), # const double *ca
        ]
        self.libtrans.normalize_compact_axang.restype = None

        self.libtrans.normalize_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q_norm
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
        ]
        self.libtrans.normalize_quat.restype = None

        # conversions.h
        # Conversions to rotation matrix
        self.libtrans.ang_to_rmat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *m
            ctypes.c_int, # const int basis
            ctypes.c_double, # const double ang
            ctypes.c_bool, # const bool passive
        ]
        self.libtrans.ang_to_rmat.restype = None

        self.libtrans.euler_to_rmat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *m
            np.ctypeslib.ndpointer(dtype=np.double), # const double *e
            ctypes.c_int, # int i
            ctypes.c_int, # int j
            ctypes.c_int, # int k
            ctypes.c_bool, # const bool extrinsic
        ]
        self.libtrans.euler_to_rmat.restype = None

        self.libtrans.axang_to_rmat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *m
            np.ctypeslib.ndpointer(dtype=np.double), # const double *a
        ]
        self.libtrans.axang_to_rmat.restype = None

        self.libtrans.compact_axang_to_rmat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *m
            np.ctypeslib.ndpointer(dtype=np.double), # const double *ca
        ]
        self.libtrans.compact_axang_to_rmat.restype = None

        self.libtrans.quat_to_rmat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *m
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
        ]
        self.libtrans.quat_to_rmat.restype = None

        # Conversions to axis angle
        self.libtrans.rmat_to_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m
        ]
        self.libtrans.rmat_to_axang.restype = None

        self.libtrans.quat_to_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
        ]
        self.libtrans.quat_to_axang.restype = None

        self.libtrans.compact_axang_to_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
            np.ctypeslib.ndpointer(dtype=np.double), # const double *ca
        ]
        self.libtrans.compact_axang_to_axang.restype = None

        self.libtrans.axang_to_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
            np.ctypeslib.ndpointer(dtype=np.double), # const double *a
        ]
        self.libtrans.axang_to_compact_axang.restype = None

        self.libtrans.rmat_to_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m
        ]
        self.libtrans.rmat_to_compact_axang.restype = None

        self.libtrans.quat_to_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
        ]
        self.libtrans.quat_to_compact_axang.restype = None

        # Conversions to quaternion
        self.libtrans.ang_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            ctypes.c_int, # const int basis
            ctypes.c_double, # const double ang
        ]
        self.libtrans.ang_to_quat.restype = None

        self.libtrans.rmat_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m
        ]
        self.libtrans.rmat_to_quat.restype = None

        self.libtrans.axang_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # const double *a
        ]
        self.libtrans.axang_to_quat.restype = None

        self.libtrans.compact_axang_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # const double *ca
        ]
        self.libtrans.compact_axang_to_quat.restype = None

        self.libtrans.euler_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # const double *e
            ctypes.c_int, # int i
            ctypes.c_int, # int j
            ctypes.c_int, # int k
            ctypes.c_bool, # const bool extrinsic
        ]
        self.libtrans.euler_to_quat.restype = None

        self.libtrans.quat_xyzw_to_wxyz.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q_wxyz
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q_xyzw
        ]
        self.libtrans.quat_xyzw_to_wxyz.restype = None

        self.libtrans.quat_wxyz_to_xyzw.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q_xyzw
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q_wxyz
        ]
        self.libtrans.quat_wxyz_to_xyzw.restype = None

        # Conversions to Euler angle
        self.libtrans.quat_to_euler.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *e
            np.ctypeslib.ndpointer(dtype=np.double), # const double *q
            ctypes.c_int, # int i
            ctypes.c_int, # int j
            ctypes.c_int, # int k
            ctypes.c_bool, # const bool extrinsic
        ]
        self.libtrans.quat_to_euler.restype = None

        self.libtrans.rmat_to_euler.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *e
            np.ctypeslib.ndpointer(dtype=np.double), # const double *m
            ctypes.c_int, # int i
            ctypes.c_int, # int j
            ctypes.c_int, # int k
            ctypes.c_bool, # const bool extrinsic
        ]
        self.libtrans.rmat_to_euler.restype = None

    # matrix.h
    def mat_multiply(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        assert m1.ndim == m2.ndim == 2
        m1, m2 = m1.astype(np.double), m2.astype(np.double)
        r1, c1 = m1.shape
        c2 = m2.shape[1]
        m12 = np.zeros(shape=(r1, c2), dtype=np.double)
        self.libtrans.mat_multiply(m12, m1, m2, r1, c1, c2)
        return m12

    def mat_transpose(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim == 2
        m = m.astype(np.double)
        r, c = m.shape
        mt = np.zeros(shape=(c, r), dtype=np.double)
        self.libtrans.mat_transpose(mt, m, r, c)
        return mt

    # vector.h
    def orthogonal_project(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        assert u.ndim == v.ndim == 1
        assert u.shape[0] == v.shape[0]
        u, v = u.astype(np.double), v.astype(np.double)
        n = u.shape[0]
        proj = np.zeros(shape=(n, ), dtype=np.double)
        self.libtrans.orthogonal_project(proj, u, v, n)
        return proj

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        assert v.ndim == 1
        v = v.astype(np.double)
        n = v.shape[0]
        v_norm = np.zeros(shape=(n, ), dtype=np.double)
        self.libtrans.normalize_vector(v_norm, v, n)
        return v_norm

    def angle_between_vectors(self, u: np.ndarray, v: np.ndarray, fast: bool = False, n: int = 3) -> float:
        '''
        NOTE: To follow the function defined in pytransform3d, 
              we set n = 3 default.
        '''
        assert u.ndim == 1 and u.shape[0] == n
        assert v.ndim == 1 and v.shape[0] == n
        u = u.astype(np.double)
        v = v.astype(np.double)
        return self.libtrans.angle_between_vectors(u, v, fast, n)

    # trans_utils.h
    # Helper functions
    def quat_prod_vec(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        assert q.ndim == 1 and v.ndim == 1
        assert q.shape[0] == 4 and v.shape[0] == 3
        q, v = q.astype(np.double), v.astype(np.double)
        qv = np.zeros(shape=(3, ), dtype=np.double)
        self.libtrans.quat_prod_vec(qv, q, v)
        return qv

    def quat_product(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        assert p.ndim == q.ndim == 1
        assert p.shape[0] == q.shape[0] == 4
        p, q = p.astype(np.double), q.astype(np.double)
        pq = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.quat_product(pq, p, q)
        return pq

    def quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        assert q.ndim == 1 and q.shape[0] == 4
        q = q.astype(np.double)
        q_conj = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.quat_conjugate(q_conj, q)
        return q_conj
    
    def pick_closest_quaternion(self, q: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        assert q.ndim == 1 and q.shape[0] == 4
        assert q_target.ndim == 1 and q_target.shape[0] == 4
        q = q.astype(np.double)
        q_target = q_target.astype(np.double)
        closest = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.pick_closest_quaternion(closest, q, q_target)
        return closest

    def quat_slerp(self, start: np.ndarray, end: np.ndarray, t: float, shortest_path: bool = False) -> np.ndarray:
        assert start.ndim == 1 and start.shape[0] == 4
        assert end.ndim == 1 and end.shape[0] == 4
        start = start.astype(np.double)
        end = end.astype(np.double)
        slerp = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.quat_slerp(slerp, start, end, t, shortest_path)
        return slerp
    
    def quat_distance(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        assert p.ndim == 1 and p.shape[0] == 4
        assert q.ndim == 1 and q.shape[0] == 4
        p = p.astype(np.double)
        q = q.astype(np.double)
        return self.libtrans.quat_distance(p, q)

    # Validation functions
    def is_rmat(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim == 2 and m.shape[0] == m.shape[1] == 3
        return self.libtrans.is_rmat(m)

    def normalize_ang(self, ang: np.double) -> np.double:
        return self.libtrans.normalize_ang(ang)

    def normalize_axang(self, a: np.ndarray) -> np.ndarray:
        assert a.ndim == 1 and a.shape[0] == 4
        a = a.astype(np.double)
        a_norm = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.normalize_axang(a_norm, a)
        return a_norm

    def normalize_compact_axang(self, ca: np.ndarray) -> np.ndarray:
        assert ca.ndim == 1 and ca.shape[0] == 3
        ca = ca.astype(np.double)
        ca_norm = np.zeros(shape=(3, ), dtype=np.double)
        self.libtrans.normalize_compact_axang(ca_norm, ca)
        return ca_norm

    def normalize_quat(self, q: np.ndarray) -> np.ndarray:
        assert q.ndim == 1 and q.shape[0] == 4
        q = q.astype(np.double)
        q_norm = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.normalize_quat(q_norm, q)
        return q_norm

    # conversions.h
    # Conversions to rotation matrix
    def ang_to_rmat(self, basis: int, ang: np.double, passive: bool) -> np.ndarray:
        assert 0 <= basis <= 2
        m = np.zeros(shape=(3, 3), dtype=np.double)
        self.libtrans.ang_to_rmat(m, basis, ang, passive)
        return m

    def euler_to_rmat(self,
                      e: np.ndarray,
                      i: int,
                      j: int,
                      k: int,
                      extrinsic: bool) -> np.ndarray:
        assert e.ndim == 1 and e.shape[0] == 3
        assert 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 2
        e = e.astype(np.double)
        m = np.zeros(shape=(3, 3), dtype=np.double)
        self.libtrans.euler_to_rmat(m, e, i, j, k, extrinsic)
        return m

    def axang_to_rmat(self, a: np.ndarray) -> np.ndarray:
        assert a.ndim == 1 and a.shape[0] == 4
        a = a.astype(np.double)
        m = np.zeros(shape=(3, 3), dtype=np.double)
        self.libtrans.axang_to_rmat(m, a)
        return m

    def compact_axang_to_rmat(self, ca: np.ndarray) -> np.ndarray:
        assert ca.ndim == 1 and ca.shape[0] == 3
        ca = ca.astype(np.double)
        m = np.zeros(shape=(3, 3), dtype=np.double)
        self.libtrans.compact_axang_to_rmat(m, ca)
        return m

    def quat_to_rmat(self, q: np.ndarray) -> np.ndarray:
        assert q.ndim == 1 and q.shape[0] == 4
        q = q.astype(np.double)
        m = np.zeros(shape=(3, 3), dtype=np.double)
        self.libtrans.quat_to_rmat(m, q)
        return m

    # Conversions to axis angle
    def rmat_to_axang(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim == 2 and m.shape[0] == m.shape[1] == 3
        a = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.rmat_to_axang(a, m)
        return a

    def quat_to_axang(self, q: np.ndarray) -> np.ndarray:
        assert q.ndim == 1 and q.shape[0] == 4
        q = q.astype(np.double)
        a = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.quat_to_axang(a, q)
        return a

    def compact_axang_to_axang(self, ca: np.ndarray) -> np.ndarray:
        assert ca.ndim == 1 and ca.shape[0] == 3
        ca = ca.astype(np.double)
        a = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.compact_axang_to_axang(a, ca)
        return a

    def axang_to_compact_axang(self, a: np.ndarray) -> np.ndarray:
        assert a.ndim == 1 and a.shape[0] == 4
        a = a.astype(np.double)
        ca = np.zeros(shape=(3, ), dtype=np.double)
        self.libtrans.axang_to_compact_axang(ca, a)
        return ca

    def rmat_to_compact_axang(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim == 2 and m.shape[0] == m.shape[1] == 3
        m = m.astype(np.double)
        ca = np.zeros(shape=(3, ), dtype=np.double)
        self.libtrans.rmat_to_compact_axang(ca, m)
        return ca

    def quat_to_compact_axang(self, q: np.ndarray) -> np.ndarray:
        assert q.ndim == 1 and q.shape[0] == 4
        q = q.astype(np.double)
        ca = np.zeros(shape=(3, ), dtype=np.double)
        self.libtrans.quat_to_compact_axang(ca, q)
        return ca

    # Conversions to quaternion
    def ang_to_quat(self, basis: int, ang: np.double) -> np.ndarray:
        q = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.ang_to_quat(q, basis, ang)
        return q

    def rmat_to_quat(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim == 2 and m.shape[0] == m.shape[1] == 3
        m = m.astype(np.double)
        q = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.rmat_to_quat(q, m)
        return q

    def axang_to_quat(self, a: np.ndarray) -> np.ndarray:
        assert a.ndim == 1 and a.shape[0] == 4
        a = a.astype(np.double)
        q = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.axang_to_quat(q, a)
        return q

    def compact_axang_to_quat(self, ca: np.ndarray) -> np.ndarray:
        assert ca.ndim == 1 and ca.shape[0] == 3
        ca = ca.astype(np.double)
        q = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.compact_axang_to_quat(q, ca)
        return q

    def euler_to_quat(self,
                      e: np.ndarray,
                      i: int,
                      j: int,
                      k: int,
                      extrinsic: bool):
        assert e.ndim == 1 and e.shape[0] == 3
        assert 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 2
        e = e.astype(np.double)
        q = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.euler_to_quat(q, e, i, j, k, extrinsic)
        return q
    
    # TODO: debug to here
    def quat_xyzw_to_wxyz(self, q_xyzw: np.ndarray) -> np.ndarray:
        assert q_xyzw.ndim == 1 and q_xyzw.shape[0] == 4
        q_xyzw = q_xyzw.astype(np.double)
        q_wxyz = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.quat_xyzw_to_wxyz(q_wxyz, q_xyzw)
        return q_wxyz

    def quat_wxyz_to_xyzw(self, q_wxyz: np.ndarray) -> np.ndarray:
        assert q_wxyz.ndim == 1 and q_wxyz.shape[0] == 4
        q_wxyz = q_wxyz.astype(np.double)
        q_xyzw = np.zeros(shape=(4, ), dtype=np.double)
        self.libtrans.quat_wxyz_to_xyzw(q_xyzw, q_wxyz)
        return q_xyzw

    # Conversions to Euler angle
    def quat_to_euler(self,
                      q: np.ndarray,
                      i: int,
                      j: int,
                      k: int,
                      extrinsic: bool) -> np.ndarray:
        assert q.ndim == 1 and q.shape[0] == 4
        assert 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 2
        q = q.astype(np.double)
        e = np.zeros((3, ), dtype=np.double)
        self.libtrans.quat_to_euler(e, q, i, j, k, extrinsic)
        return e

    def rmat_to_euler(self,
                      m: np.ndarray,
                      i: int,
                      j: int,
                      k: int,
                      extrinsic: bool) -> np.ndarray:
        assert m.ndim == 2 and m.shape[0] == m.shape[1] == 3
        assert 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 2
        m = m.astype(np.double)
        e = np.zeros((3, ), dtype=np.double)
        self.libtrans.rmat_to_euler(e, m, i, j, k, extrinsic)
        return e
