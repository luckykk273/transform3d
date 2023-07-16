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


class LibWrapper:
    '''
    All C library is wrapped in this class.
    '''
    def __init__(self, libpath: str = '../build/libtransformation.dll') -> None:
        self.libtrans = ctypes.cdll.LoadLibrary(libpath)

        # necessary for clib
        self.c_double_ptr = ctypes.POINTER(ctypes.c_double)  # double *p
        self.c_double_3_array = ctypes.c_double * 3  # double arr[3]
        self.c_array_type = self.c_double_3_array * 3
        self.c_2d_array_ptr = ctypes.POINTER(self.c_array_type)  # double (*arr)[3]

        # define arg types and res type
        # utils.h
        # Helper functions
        self.libtrans.mat33_dot.argtypes = [
            self.c_2d_array_ptr, # double (*m1)[3]
            self.c_2d_array_ptr, # double (*m2)[3]
            self.c_2d_array_ptr, # double (*m12)[3]
        ]
        self.libtrans.mat33_dot.restype = None

        self.libtrans.mat33_transpose.argtypes = [
            self.c_2d_array_ptr, # double (*m)[3]
            self.c_2d_array_ptr, # double (*mT)[3]
        ]
        self.libtrans.mat33_transpose.restype = None

        self.libtrans.vec_proj.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
            np.ctypeslib.ndpointer(dtype=np.double), # double *b
            np.ctypeslib.ndpointer(dtype=np.double), # double *a_on_b
        ]
        self.libtrans.vec_proj.restype = None

        self.libtrans.quat_prod_vec.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # double *v
            np.ctypeslib.ndpointer(dtype=np.double), # double *v_res
        ]
        self.libtrans.quat_prod_vec.restype = None

        self.libtrans.quat_mul.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q0
            np.ctypeslib.ndpointer(dtype=np.double), # double *q1
            np.ctypeslib.ndpointer(dtype=np.double), # double *q01
        ]
        self.libtrans.quat_mul.restype = None

        self.libtrans.quat_conj.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # double *q_conj
        ]
        self.libtrans.quat_conj.restype = None

        # Validation functions
        self.libtrans.is_rmat.argtypes = [
            self.c_2d_array_ptr, # double (*m)[3]
        ]
        self.libtrans.is_rmat.restype = ctypes.c_bool

        self.libtrans.normalize_ang.argtypes = [
            ctypes.c_double, # double ang
        ]
        self.libtrans.normalize_ang.restype = ctypes.c_double

        self.libtrans.normalize_vec.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *v
        ]
        self.libtrans.normalize_vec.restype = None

        self.libtrans.normalize_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
        ]
        self.libtrans.normalize_axang.restype = None

        self.libtrans.normalize_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
        ]
        self.libtrans.normalize_compact_axang.restype = None

        self.libtrans.normalize_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
        ]
        self.libtrans.normalize_quat.restype = None

        # conversions.h
        # Conversions to rotation matrix
        self.libtrans.ang_to_rmat.argtypes = [
            self.c_2d_array_ptr, # double (*m)[3]
            ctypes.c_int32, # int basis
            ctypes.c_double, # double ang
            ctypes.c_bool, # bool passive
        ]
        self.libtrans.ang_to_rmat.restype = None

        self.libtrans.euler_to_rmat.argtypes = [
            self.c_2d_array_ptr, # double (*m)[3]
            np.ctypeslib.ndpointer(dtype=np.double), # double *e
            ctypes.c_int32, # int i
            ctypes.c_int32, # int j
            ctypes.c_int32, # int k
            ctypes.c_bool, # bool extrinsic
        ]
        self.libtrans.euler_to_rmat.restype = None

        self.libtrans.axang_to_rmat.argtypes = [
            self.c_2d_array_ptr, # double (*m)[3]
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
        ]
        self.libtrans.axang_to_rmat.restype = None

        self.libtrans.compact_axang_to_rmat.argtypes = [
            self.c_2d_array_ptr, # double (*m)[3]
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
        ]
        self.libtrans.compact_axang_to_rmat.restype = None

        self.libtrans.quat_to_rmat.argtypes = [
            self.c_2d_array_ptr, # double (*m)[3]
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
        ]
        self.libtrans.quat_to_rmat.restype = None

        # Conversions to axis angle
        self.libtrans.rmat_to_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
            self.c_2d_array_ptr, # double (*m)[3]
        ]
        self.libtrans.rmat_to_axang.restype = None

        self.libtrans.quat_to_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
        ]
        self.libtrans.quat_to_axang.restype = None

        self.libtrans.compact_axang_to_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
        ]
        self.libtrans.compact_axang_to_axang.restype = None

        self.libtrans.axang_to_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
        ]
        self.libtrans.axang_to_compact_axang.restype = None

        self.libtrans.rmat_to_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
            self.c_2d_array_ptr, # double (*m)[3]
        ]
        self.libtrans.rmat_to_compact_axang.restype = None

        self.libtrans.quat_to_compact_axang.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
        ]
        self.libtrans.quat_to_compact_axang.restype = None

        # Conversions to quaternion
        self.libtrans.ang_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            ctypes.c_int32, # int basis
            ctypes.c_double, # double ang
        ]
        self.libtrans.ang_to_quat.restype = None

        self.libtrans.rmat_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            self.c_2d_array_ptr, # double (*m)[3]
        ]
        self.libtrans.rmat_to_quat.restype = None

        self.libtrans.axang_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # double *a
        ]
        self.libtrans.axang_to_quat.restype = None

        self.libtrans.compact_axang_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # double *ca
        ]
        self.libtrans.compact_axang_to_quat.restype = None

        self.libtrans.euler_to_quat.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            np.ctypeslib.ndpointer(dtype=np.double), # double *e
            ctypes.c_int32, # int i
            ctypes.c_int32, # int j
            ctypes.c_int32, # int k
            ctypes.c_bool, # bool extrinsic
        ]
        self.libtrans.euler_to_quat.restype = None

        self.libtrans.quat_xyzw_to_wxyz.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
        ]
        self.libtrans.quat_xyzw_to_wxyz.restype = None

        self.libtrans.quat_wxyz_to_xyzw.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
        ]
        self.libtrans.quat_wxyz_to_xyzw.restype = None

        # Conversions to Euler angle
        self.libtrans.quat_to_euler.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *e
            np.ctypeslib.ndpointer(dtype=np.double), # double *q
            ctypes.c_int32, # int i
            ctypes.c_int32, # int j
            ctypes.c_int32, # int k
            ctypes.c_bool, # bool extrinsic
        ]
        self.libtrans.quat_to_euler.restype = None

        self.libtrans.rmat_to_euler.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double), # double *e
            self.c_2d_array_ptr, # double (*m)[3]
            ctypes.c_int32, # int i
            ctypes.c_int32, # int j
            ctypes.c_int32, # int k
            ctypes.c_bool, # bool extrinsic
        ]
        self.libtrans.rmat_to_euler.restype = None

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

    # utils.h
    # Helper functions
    def mat33_dot(self, m1: np.ndarray, m2: np.ndarray):
        m1, m2 = m1.astype(np.double), m2.astype(np.double)
        c_m1 = self.np_2d_array_to_c_2d_array(m1)
        c_m2 = self.np_2d_array_to_c_2d_array(m2)
        c_m12 = self._get_c_2d_array()
        self.libtrans.mat33_dot(
            ctypes.byref(c_m1),
            ctypes.byref(c_m2),
            ctypes.byref(c_m12)
        )
        return self.c_2d_array_to_np_2d_array(c_m12)

    def mat33_transpose(self, m: np.ndarray):
        m = m.astype(np.double)
        c_m = self.np_2d_array_to_c_2d_array(m)
        c_mT = self._get_c_2d_array()
        self.libtrans.mat33_transpose(
            ctypes.byref(c_m),
            ctypes.byref(c_mT)
        )
        return self.c_2d_array_to_np_2d_array(c_mT)

    def vec_proj(self, a: np.ndarray, b: np.ndarray):
        a, b = a.astype(np.double), b.astype(np.double)
        a_on_b = np.zeros((3, ), dtype=np.double)
        self.libtrans.vec_proj(a, b, a_on_b)
        return a_on_b

    def quat_prod_vec(self, q: np.ndarray, v: np.ndarray):
        q, v = q.astype(np.double), v.astype(np.double)
        v_res = np.zeros((3, ), dtype=np.double)
        self.libtrans.quat_prod_vec(q, v, v_res)
        return v_res

    def quat_mul(self, q0: np.ndarray, q1: np.ndarray):
        q0, q1 = q0.astype(np.double), q1.astype(np.double)
        q01 = np.zeros((4, ), dtype=np.double)
        self.libtrans.quat_mul(q0, q1, q01)
        return q01

    def quat_conj(self, q: np.ndarray):
        q = q.astype(np.double)
        q_conj = np.zeros((4, ), dtype=np.double)
        self.libtrans.quat_conj(q, q_conj)
        return q_conj

    def is_rmat(self, m: np.ndarray):
        c_m = self.np_2d_array_to_c_2d_array(m)
        return self.libtrans.is_rmat(ctypes.byref(c_m))

    def normalize_ang(self, ang: np.double):
        return self.libtrans.normalize_ang(ang)

    def normalize_vec(self, v: np.ndarray):
        v = v.astype(np.double)
        self.libtrans.normalize_vec(v)
        return v

    def normalize_axang(self, a: np.ndarray):
        a = a.astype(np.double)
        self.libtrans.normalize_axang(a)
        return a

    def normalize_compact_axang(self, ca: np.ndarray):
        ca = ca.astype(np.double)
        self.libtrans.normalize_compact_axang(ca)
        return ca

    def normalize_quat(self, q: np.ndarray):
        q = q.astype(np.double)
        self.libtrans.normalize_quat(q)
        return q

    # conversions.h
    # Conversions to rotation matrix
    def ang_to_rmat(self, basis: np.int32, ang: np.double, passive: bool):
        c_m = self._get_c_2d_array()
        self.libtrans.ang_to_rmat(ctypes.byref(c_m), basis, ang, passive)
        return self.c_2d_array_to_np_2d_array(c_m)

    def euler_to_rmat(self,
                      e: np.ndarray,
                      i: np.int32,
                      j: np.int32,
                      k: np.int32,
                      extrinsic: bool):
        c_m = self._get_c_2d_array()
        e = e.astype(np.double)
        self.libtrans.euler_to_rmat(ctypes.byref(c_m), e, i, j, k, extrinsic)
        return self.c_2d_array_to_np_2d_array(c_m)

    def axang_to_rmat(self, a: np.ndarray):
        c_m = self._get_c_2d_array()
        a = a.astype(np.double)
        self.libtrans.axang_to_rmat(ctypes.byref(c_m), a)
        return self.c_2d_array_to_np_2d_array(c_m)

    def compact_axang_to_rmat(self, ca: np.ndarray):
        c_m = self._get_c_2d_array()
        ca = ca.astype(np.double)
        self.libtrans.compact_axang_to_rmat(ctypes.byref(c_m), ca)
        return self.c_2d_array_to_np_2d_array(c_m)

    def quat_to_rmat(self, q: np.ndarray):
        c_m = self._get_c_2d_array()
        q = q.astype(np.double)
        self.libtrans.quat_to_rmat(ctypes.byref(c_m), q)
        return self.c_2d_array_to_np_2d_array(c_m)

    # Conversions to axis angle
    def rmat_to_axang(self, m: np.ndarray):
        c_m = self.np_2d_array_to_c_2d_array(m)
        a = np.zeros((4, ), dtype=np.double)
        self.libtrans.rmat_to_axang(a, ctypes.byref(c_m))
        return a

    def quat_to_axang(self, q: np.ndarray):
        a = np.zeros((4, ), dtype=np.double)
        q = q.astype(np.double)
        self.libtrans.quat_to_axang(a, q)
        return a

    def compact_axang_to_axang(self, ca: np.ndarray):
        a = np.zeros((4, ), dtype=np.double)
        ca = ca.astype(np.double)
        self.libtrans.compact_axang_to_axang(a, ca)
        return a

    def axang_to_compact_axang(self, a: np.ndarray):
        ca = np.zeros((3, ), dtype=np.double)
        a = a.astype(np.double)
        self.libtrans.axang_to_compact_axang(ca, a)
        return ca

    def rmat_to_compact_axang(self, m: np.ndarray):
        ca = np.zeros((3, ), dtype=np.double)
        c_m = self.np_2d_array_to_c_2d_array(m)
        self.libtrans.rmat_to_compact_axang(ca, ctypes.byref(c_m))
        return ca

    def quat_to_compact_axang(self, q: np.ndarray):
        ca = np.zeros((3, ), dtype=np.double)
        q = q.astype(np.double)
        self.libtrans.quat_to_compact_axang(ca, q)
        return ca

    # Conversions to quaternion
    def ang_to_quat(self, basis: np.int32, ang: np.double):
        q = np.zeros((4, ), dtype=np.double)
        self.libtrans.ang_to_quat(q, basis, ang)
        return q

    def rmat_to_quat(self, m: np.ndarray):
        q = np.zeros((4, ), dtype=np.double)
        c_m = self.np_2d_array_to_c_2d_array(m)
        self.libtrans.rmat_to_quat(q, ctypes.byref(c_m))
        return q

    def axang_to_quat(self, a: np.ndarray):
        q = np.zeros((4, ), dtype=np.double)
        a = a.astype(np.double)
        self.libtrans.axang_to_quat(q, a)
        return q

    def compact_axang_to_quat(self, ca: np.ndarray):
        q = np.zeros((4, ), dtype=np.double)
        ca = ca.astype(np.double)
        self.libtrans.compact_axang_to_quat(q, ca)
        return q

    def euler_to_quat(self,
                      e: np.ndarray,
                      i: np.int32,
                      j: np.int32,
                      k: np.int32,
                      extrinsic: bool):
        q = np.zeros((4, ), dtype=np.double)
        e = e.astype(np.double)
        self.libtrans.euler_to_quat(q, e, i, j, k, extrinsic)
        return q

    def quat_xyzw_to_wxyz(self, q: np.ndarray):
        q = q.astype(np.double)
        self.libtrans.quat_xyzw_to_wxyz(q)
        return q

    def quat_wxyz_to_xyzw(self, q: np.ndarray):
        q = q.astype(np.double)
        self.libtrans.quat_wxyz_to_xyzw(q)
        return q

    # Conversions to Euler angle
    def quat_to_euler(self,
                      q: np.ndarray,
                      i: np.int32,
                      j: np.int32,
                      k: np.int32,
                      extrinsic: bool):
        e = np.zeros((3, ), dtype=np.double)
        q = q.astype(np.double)
        self.libtrans.quat_to_euler(e, q, i, j, k, extrinsic)
        return e

    def rmat_to_euler(self,
                      m: np.ndarray,
                      i: np.int32,
                      j: np.int32,
                      k: np.int32,
                      extrinsic: bool):
        e = np.zeros((3, ), dtype=np.double)
        c_m = self.np_2d_array_to_c_2d_array(m)
        self.libtrans.rmat_to_euler(e, ctypes.byref(c_m), i, j, k, extrinsic)
        return e
