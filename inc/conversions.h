#ifndef TRANS_CONVERSIONS_H_
#define TRANS_CONVERSIONS_H_

#include <stdbool.h>

#define CONVERSIONS_TYPE_CHECK  (1U)

#ifdef __cplusplus
extern "C" {
#endif


// Note: Euler sequence, Euler order, Euler convention, are all general names to describe
// the representation of the three consecutive rotations.
// Here we choose the term `order`.
enum EulerOrderE {
  // Cardan (or Tait-Bryan) angles
  EULER_ORDER_XYZ = 0,
  EULER_ORDER_XZY,
  EULER_ORDER_YXZ,
  EULER_ORDER_YZX,
  EULER_ORDER_ZXY,
  EULER_ORDER_ZYX,
  // proper Euler angles
  EULER_ORDER_XYX,
  EULER_ORDER_XZX,
  EULER_ORDER_YXY,
  EULER_ORDER_YZY,
  EULER_ORDER_ZXZ,
  EULER_ORDER_ZYZ
};

/*
 * Conversions to rotation matrix
 */
// Compute passive/active rotation matrix from rotation about basis vector
void ang_to_rmat(double *m, const int basis, const double ang, const bool passive);

void euler_to_rmat(double *m, const double *e, int i, int j, int k, const bool extrinsic);

void axang_to_rmat(double *m, const double *a);

void compact_axang_to_rmat(double *m, const double *ca);

void quat_to_rmat(double *m, const double *q);

/*
 * Conversions to axis angle
 */
void rmat_to_axang(double *a, const double *m);

void quat_to_axang(double *a, const double *q);

void compact_axang_to_axang(double *a, const double *ca);

void axang_to_compact_axang(double *ca, const double *a);

void rmat_to_compact_axang(double *ca, const double *m);

void quat_to_compact_axang(double *ca, const double *q);

/*
 * Conversions to quaternion
 */
void ang_to_quat(double *q, const int basis, const double ang);

// NOTE: When computing a quaternion from the rotation matrix there is a sign
//       ambiguity: q and -q represent the same rotation.
void rmat_to_quat(double *q, const double *m);

void axang_to_quat(double *q, const double *a);

void compact_axang_to_quat(double *q, const double *ca);

void euler_to_quat(double *q, const double *e, int i, int j, int k, const bool extrinsic);

void quat_xyzw_to_wxyz(double *q_wxyz, const double *q_xyzw);

void quat_wxyz_to_xyzw(double *q_xyzw, const double *q_wxyz);

/*
 * Conversions to Euler angle
 */
// ref: https://doi.org/10.1371/journal.pone.0276302
void quat_to_euler(double *e, const double *q, int i, int j, int k, const bool extrinsic);

// ref: https://arc.aiaa.org/doi/abs/10.2514/1.16622
void rmat_to_euler(double *e, const double *m, int i, int j, int k, const bool extrinsic);

#ifdef __cplusplus
}
#endif

#endif  // TRANS_CONVERSIONS_H_