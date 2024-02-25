#include "trans_utils.h"
#include "vector.h"
#include "matrix.h"
#include "linalg.h"
#include "conversions.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <string.h>
#include <limits.h>


/*
 * Helper functions
 */
double clip(const double x, const double min, const double max) {
  // return (x < min) ? min : (x > max ? max : x);
  if (x < min) {
    return min;
  }
  if (x > max) {
    return max;
  }
  return x;
}

int sign(const double x) {
  if (x < 0.0) {
    return -1;
  }
  if (x > 0.0) {
    return 1;
  }
  return 0;
}

bool is_bit_zero(double x) {
  return *((int64_t*) &x) == 0;
}

uint64_t factorial(uint64_t n) {
  uint64_t res = 1;
  uint64_t i;
  // Check for overflow at each step
  for (i = 1; i <= n; ++i) {
      // Check for potential overflow before multiplying
      if (UINT64_MAX / res < i) {
          return 0;  // Return an error value
      }
      res *= i;
  }

  return res;
}

void swap(double *a, double *b) {
  double tmp = *a;
  *a = *b;
  *b = tmp;
}

double q_rsqrt(const double number) {
  union {
    double   f;
    uint64_t i;
  } conv = { .f = number };
  conv.i  = 0x5fe6ec85e7de30daL - (conv.i >> 1);  // what the fuck?
  conv.f *= 1.5 - (number * 0.5 * conv.f * conv.f);  // 1st iteration
  // conv.f *= 1.5 - (number * 0.5 * conv.f * conv.f);  // 2nd iteration, this can be removed
  return conv.f;
}

void quat_prod_vec(double *qv, const double *q, const double *v) {
  double t[3], cross_qt[3];
  cross_product(t, &q[1], v);
  size_t i;
  for (i = 0; i < 3; ++i) {
    t[i] *= 2.0;
  }

  cross_product(cross_qt, &q[1], t);
  for (i = 0; i < 3; ++i) {
    qv[i] = v[i] + q[0] * t[i] + cross_qt[i];
  }
}

void quat_product(double *pq, const double *p, const double *q) {
  pq[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
  pq[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
  pq[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
  pq[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
}

void quat_conjugate(double *q_conj, const double *q) {
  q_conj[0] = q[0];
  q_conj[1] = -q[1];
  q_conj[2] = -q[2];
  q_conj[3] = -q[3];
}

void pick_closest_quaternion(double *closest, const double *q, const double *q_target) {
  double q_norm[4], q_target_norm[4];
  normalize_quat(q_norm, q);
  normalize_quat(q_target_norm, q_target);

  double norm_neg = 0.0, norm_pos = 0.0;
  size_t i;
  for(i = 0; i < 4; ++i) {
    norm_neg += (-q[i] - q_target[i]) * ((-q[i] - q_target[i]));
    norm_pos += (q[i] - q_target[i]) * (q[i] - q_target[i]);
  }
  norm_neg = sqrt(norm_neg);
  norm_pos = sqrt(norm_pos);
  double sign = (norm_neg < norm_pos) ? -1.0 : 1.0;
  for(i = 0; i < 4; ++i) {
    closest[i] = q[i] * sign;
  }
}

void quat_lerp(double *lerp, const double *start, const double *end, const double t, const bool shortest_path) {
  double start_norm[4], end_norm[4];
  normalize_quat(start_norm, start);
  normalize_quat(end_norm, end);
  if(shortest_path) {
    pick_closest_quaternion(end_norm, end, start);
  }
  vec_lerp(lerp, start_norm, end_norm, t, 4);
  normalize_quat(lerp, lerp);
}

void quat_slerp(double *slerp, const double *start, const double *end, const double t, const bool shortest_path) {
  double start_norm[4], end_norm[4];
  normalize_quat(start_norm, start);
  normalize_quat(end_norm, end);
  if(shortest_path) {
    pick_closest_quaternion(end_norm, end, start);
  }
  vec_slerp(slerp, start_norm, end_norm, t, 4);
}

double quat_distance(const double *p, const double *q) {
  double p_norm[4], q_norm[4];
  normalize_quat(p_norm, p);
  normalize_quat(q_norm, q);

  double q_conj[4], pq_conj[4];
  quat_conjugate(q_conj, q);
  quat_product(pq_conj, p, q_conj);
  
  double axang[4];
  quat_to_axang(axang, pq_conj);
  return fmin(axang[3], 2.0 * M_PI - axang[3]);
}

void omega(double *omega, const double *w) {
  omega[0] = 0.0;
  omega[1] = -w[0];
  omega[2] = -w[1];
  omega[3] = -w[2];

  omega[4] = w[0];
  omega[5] = 0.0;
  omega[6] = w[2];
  omega[7] = -w[1];

  omega[8] = w[1];
  omega[9] = -w[2];
  omega[10] = 0.0;
  omega[11] = w[0];

  omega[12] = w[2];
  omega[13] = w[1];
  omega[14] = -w[0];
  omega[15] = 0.0;
}

/*
 * Validation functions
 */
bool is_rmat(const double *m) {
  double mt[3][3], mmt[3][3];
  mat_transpose(&mt[0][0], m, 3, 3);
  mat_multiply(&mmt[0][0], m, &mt[0][0], 3, 3, 3);
  double eps = 1e-6;
  double e;
  size_t i, j;
  for(i = 0; i < 3; ++i) {
    for(j = 0; j < 3; ++j) {
      e = (i == j) ? 1.0 : 0.0;
      if(fabs(mmt[i][j] - e) > eps) {
        return false;
      }
    }
  }

  double m_cpy[3][3];
  memcpy(&m_cpy[0][0], m, 9 * sizeof(double));
  return (fabs(determinant(&m_cpy[0][0], 3) - 1.0) < eps);
}

double normalize_ang(const double ang) {
  double TWO_PI = 2.0 * M_PI;
  return (ang - (ceil((ang + M_PI) / TWO_PI) - 1.0) * TWO_PI);  // (-Pi, Pi]
  // return (ang - floor((ang + M_PI) / TWO_PI) * TWO_PI);  // [-Pi, Pi)
}

void normalize_axang(double *a_norm, const double *a) {
  double ang = a[3];
  double norm = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  if (fabs(ang) < __DBL_EPSILON__ || norm < __DBL_EPSILON__) {
    a_norm[0] = 1.0;
    a_norm[1] = 0.0;
    a_norm[2] = 0.0;
    a_norm[3] = 0.0;
    return;
  }

  double reci_norm = 1.0 / norm;
  a_norm[0] = a[0] * reci_norm;
  a_norm[1] = a[1] * reci_norm;
  a_norm[2] = a[2] * reci_norm;
  ang = normalize_ang(ang);
  if (ang < 0.0) {
    ang = -ang;
    a_norm[0] = -a[0];
    a_norm[1] = -a[1];
    a_norm[2] = -a[2];
  }

  a_norm[3] = ang;
}

void normalize_compact_axang(double *ca_norm, const double *ca) {
  double eps = 1e-6;
  double ang = sqrt(ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2]);
  if (ang < eps) {
    ca_norm[0] = 0.0;
    ca_norm[1] = 0.0;
    ca_norm[2] = 0.0;
    return;
  }

  double ang_inv = 1.0 / ang;
  ca_norm[0] = ca[0] * ang_inv;
  ca_norm[1] = ca[1] * ang_inv;
  ca_norm[2] = ca[2] * ang_inv;
  ang = normalize_ang(ang);
  ca_norm[0] *= ang;
  ca_norm[1] *= ang;
  ca_norm[2] *= ang;
}

void normalize_quat(double *q_norm, const double *q) {
  double eps = 1e-6;
  double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm < eps) {
    q_norm[0] = q[0];
    q_norm[1] = q[1];
    q_norm[2] = q[2];
    q_norm[3] = q[3];
    return;
  }

  double reci_norm = 1.0 / norm;
  q_norm[0] = q[0] * reci_norm;
  q_norm[1] = q[1] * reci_norm;
  q_norm[2] = q[2] * reci_norm;
  q_norm[3] = q[3] * reci_norm;
}
