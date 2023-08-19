#ifndef CONVERSIONS_H_

#define CONVERSIONS_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Conversions to rotation matrix
 */
// Compute passive/active rotation matrix from rotation about basis vector
void ang_to_rmat(double (*m)[3], int basis, double ang, bool passive);

void euler_to_rmat(double (*m)[3], double *e, int i, int j, int k,
                   bool extrinsic);

// void two_vec_to_rmat(double (*m)[3], double *a, double *b);

void axang_to_rmat(double (*m)[3], double *a);

void compact_axang_to_rmat(double (*m)[3], double *ca);

void quat_to_rmat(double (*m)[3], double *q);

/*
 * Conversions to axis angle
 */
void rmat_to_axang(double *a, double (*m)[3]);

void quat_to_axang(double *a, double *q);

void compact_axang_to_axang(double *a, double *ca);

void axang_to_compact_axang(double *ca, double *a);

void rmat_to_compact_axang(double *ca, double (*m)[3]);

void quat_to_compact_axang(double *ca, double *q);

/*
 * Conversions to quaternion
 */
void ang_to_quat(double *q, int basis, double ang);

// NOTE: When computing a quaternion from the rotation matrix there is a sign
//       ambiguity: q and -q represent the same rotation.
void rmat_to_quat(double *q, double (*m)[3]);

void axang_to_quat(double *q, double *a);

void compact_axang_to_quat(double *q, double *ca);

void euler_to_quat(double *q, double *e, int i, int j, int k, bool extrinsic);

// void mrp_to_quat(double *q, double *mrp);

void quat_xyzw_to_wxyz(double *q);

void quat_wxyz_to_xyzw(double *q);

/*
 * Conversions to Euler angle
 */
// ref: https://doi.org/10.1371/journal.pone.0276302
void quat_to_euler(double *e, double *q, int i, int j, int k, bool extrinsic);

// ref: https://arc.aiaa.org/doi/abs/10.2514/1.16622
void rmat_to_euler(double *e, double (*m)[3], int i, int j, int k, bool extrinsic);

#ifdef __cplusplus
}
#endif

#endif  // CONVERSIONS_H_