#ifndef TRANS_UTILS_H_
#define TRANS_UTILS_H_

#include <stdbool.h>
#include <stdint.h>

#define UTILS_FAST_INV_SQRT  (0U)
#define GRAVITY (9.80665)

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Helper functions
 */
double clip(const double x, const double min, const double max);

int sign(const double x);

bool is_bit_zero(double x);

// Use int to limit its maximum range.
uint64_t factorial(uint64_t n);

void swap(double *a, double *b);

// https://en.wikipedia.org/wiki/Fast_inverse_square_root
// https://stackoverflow.com/questions/11513344/how-to-implement-the-fast-inverse-square-root-in-java
double q_rsqrt(const double number);

void quat_prod_vec(double *qv, const double *q, const double *v);

// Hamilton quaternion product.
void quat_product(double *pq, const double *p, const double *q);

void quat_conjugate(double *q_conj, const double *q);

void pick_closest_quaternion(double *closest, const double *q, const double *q_target);

void quat_lerp(double *lerp, const double *start, const double *end, const double t, const bool shortest_path);

void quat_slerp(double *slerp, const double *start, const double *end, const double t, const bool shortest_path);

// Use the angular metric of S^3.
double quat_distance(const double *p, const double *q);

void omega(double *omega, const double *w);

/*
 * Validation functions
 */
// assume m is a 3x3 matrix.
// 1. (R^(-1)) R = (R^T) R = I
// 2. det(R) = 1(a pure rotation matrix does not contain any scaling factor or
// reflection, so no need to check -1)
bool is_rmat(const double *m);

// normalize an angle between -π and +π;
// default is (-π, +π];
// ref: https://stackoverflow.com/a/22949941/13932554
// NOTE: If one wants to make interval around a center value, 
//       the following reference may help.
// ref: https://commons.apache.org/proper/commons-math/javadocs/api-3.6.1/org/apache/commons/math3/util/MathUtils.html
double normalize_ang(const double ang);

// axis of rotation and rotation angle: (x, y, z, angle)
// The length of the axis vector is 1 and the angle is in [0, pi).
// No rotation is represented by [1, 0, 0, 0].
void normalize_axang(double *a_norm, const double *a);

// axis of rotation and rotation angle: angle * (x, y, z)
void normalize_compact_axang(double *ca_norm, const double *ca);

// Normalize the quaternion so that it is a unit quaternion
void normalize_quat(double *q_norm, const double *q);

#ifdef __cplusplus
}
#endif

#endif  // TRANS_UTILS_H_