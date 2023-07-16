#include <stdbool.h>

/*
 * Helper functions
 */
// assume m is a 3x3 matrix.
double determinant(double (*m)[3]);

double clip(double x, double min, double max);

int sign(double x);

double mat33_trace(double (*m)[3]);

void mat33_dot(double (*m1)[3], double (*m2)[3], double (*m12)[3]);

void mat33_transpose(double (*m)[3], double (*mT)[3]);

double vec_dot(double *a, double *b);

void vec_cross(double *v1, double *v2, double *v_res);

// Orthogonal projection of vector a on vector b.
void vec_proj(double *a, double *b, double *a_on_b);

void quat_prod_vec(double *q, double *v, double *v_res);

void quat_mul(double *q0, double *q1, double *q01);

void quat_conj(double *q, double *q_conj);

/*
 * Validation functions
 */
// assume m is a 3x3 matrix.
// 1. (R^(-1)) R = (R^T) R = I
// 2. det(R) = 1(a pure rotation matrix does not contain any scaling factor or
// reflection, so no need to check -1)
bool is_rmat(double (*m)[3]);

// normalize an angle between -π and +π;
// default is (-π, +π];
// ref: https://stackoverflow.com/a/22949941/13932554
// NOTE: If one wants to make interval around a center value, 
//       the following reference may help.
// ref: https://commons.apache.org/proper/commons-math/javadocs/api-3.6.1/org/apache/commons/math3/util/MathUtils.html
double normalize_ang(double ang);

void normalize_vec(double *v);

// axis of rotation and rotation angle: (x, y, z, angle)
// The length of the axis vector is 1 and the angle is in [0, pi).
// No rotation is represented by [1, 0, 0, 0].
void normalize_axang(double *a);

// axis of rotation and rotation angle: angle * (x, y, z)
void normalize_compact_axang(double *ca);

// Normalize the quaternion so that it is a unit quaternion
void normalize_quat(double *q);
