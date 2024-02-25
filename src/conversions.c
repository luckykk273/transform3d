#include "conversions.h"
#include "trans_utils.h"
#include "matrix.h"
#include "vector.h"
#include <assert.h>

#define _USE_MATH_DEFINES
#include <math.h>

/*
 * Conversions to rotation matrix
 */
void ang_to_rmat(double *m, const int basis, const double ang, const bool passive) {
#if CONVERSIONS_TYPE_CHECK
  assert(basis == 0 || basis == 1 || basis == 2);
#endif
  double c = cos(ang);
  double s = sin(ang);
  double sign = passive ? 1.0 : -1.0;
  s *= sign;
  if(basis == 0) {
    m[0] = 1.0, m[1] = 0.0, m[2] = 0.0;
    m[3] = 0.0, m[4] = c, m[5] = s;
    m[6] = 0.0, m[7] = -s, m[8] = c;
  } else if(basis == 1) {
    m[0] = c, m[1] = 0.0, m[2] = -s;
    m[3] = 0.0, m[4] = 1.0, m[5] = 0.0;
    m[6] = s, m[7] = 0.0, m[8] = c;
  } else if(basis == 2) {
    m[0] = c, m[1] = s, m[2] = 0.0;
    m[3] = -s, m[4] = c, m[5] = 0.0;
    m[6] = 0.0, m[7] = 0.0, m[8] = 1.0;
  }
}

void euler_to_rmat(double *m, const double *e, int i, int j, int k, const bool extrinsic) {
#if CONVERSIONS_TYPE_CHECK
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
#endif
  double alpha = e[0];
  double beta = e[1];
  double gamma = e[2];
  double m0[3][3], m1[3][3], m2[3][3], m10[3][3];
  if(!extrinsic) {
    i ^= k;
    k ^= i;
    i ^= k;
    double temp;
    temp = alpha;
    alpha = gamma;
    gamma = temp;
  }

  ang_to_rmat(&m0[0][0], i, alpha, false);
  ang_to_rmat(&m1[0][0], j, beta, false);
  ang_to_rmat(&m2[0][0], k, gamma, false);
  mat_multiply(&m10[0][0], &m1[0][0], &m0[0][0], 3, 3, 3);
  mat_multiply(m, &m2[0][0], &m10[0][0], 3, 3, 3);
}

void axang_to_rmat(double *m, const double *a) {
  double a_norm[4];
  normalize_axang(a_norm, a);
  double ax = a_norm[0], ay = a_norm[1], az = a_norm[2], theta = a_norm[3];
  double c = cos(theta), s = sin(theta), t = 1.0 - c;
  double tmp1, tmp2;
  m[0] = c + ax * ax * t;
  m[4] = c + ay * ay * t;
  m[8] = c + az * az * t;

  tmp1 = ax * ay * t;
  tmp2 = az * s;
  m[3] = tmp1 + tmp2;
  m[1] = tmp1 - tmp2;

  tmp1 = ax * az * t;
  tmp2 = ay * s;
  m[6] = tmp1 - tmp2;
  m[2] = tmp1 + tmp2;

  tmp1 = ay * az * t;
  tmp2 = ax * s;
  m[7] = tmp1 + tmp2;
  m[5] = tmp1 - tmp2;
}

void compact_axang_to_rmat(double *m, const double *ca) {
  double a[3];
  compact_axang_to_axang(a, ca);
  axang_to_rmat(m, a);
}

void quat_to_rmat(double *m, const double *q) {
  double q_norm[4];
  normalize_quat(q_norm, q);
  double w = q_norm[0], x = q_norm[1], y = q_norm[2], z = q_norm[3];
  double x2 = x + x, y2 = y + y, z2 = z + z;
  double xx = x * x2, xy = x * y2, xz = x * z2;
  double yy = y * y2, yz = y * z2, zz = z * z2;
  double wx = w * x2, wy = w * y2, wz = w * z2;
  m[0] = 1.0 - (yy + zz);
  m[1] = xy - wz;
  m[2] = xz + wy;
  m[3] = xy + wz;
  m[4] = 1.0 - (xx + zz);
  m[5] = yz - wx;
  m[6] = xz - wy;
  m[7] = yz + wx;
  m[8] = 1.0 - (xx + yy);
}

/*
 * Conversions to axis angle
 */
void rmat_to_axang(double *a, const double *m) {
#if CONVERSIONS_TYPE_CHECK
  assert(is_rmat(m));
#endif
  double cos_ang = (mat_trace(m, 3) - 1.0) / 2.0;
  double ang = acos(fmin(fmax(-1.0, cos_ang), 1.0));
  if (fabs(ang) < __DBL_EPSILON__) {
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    return;
  }
  double ax_unnormalized[3];
  ax_unnormalized[0] = m[7] - m[5];
  ax_unnormalized[1] = m[2] - m[6];
  ax_unnormalized[2] = m[3] - m[1];
  if (fabs(ang - M_PI) < 1e-4) {
    double m_diag[3], eeT_diag[3];
    double scalar;
    int i;
    for (i = 0; i < 3; ++i) {
      m_diag[i] = clip(m[i * 3 + i], -1.0, 1.0);
      eeT_diag[i] = 0.5 * (m_diag[i] + 1.0);
      scalar = (sign(ax_unnormalized[i]) == -1) ? -1.0 : 1.0;
      a[i] = sqrt(eeT_diag[i]) * scalar;
    }
  } else {
    a[0] = ax_unnormalized[0];
    a[1] = ax_unnormalized[1];
    a[2] = ax_unnormalized[2];
  }
  
  double norm = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  double reci_norm;
#if UTILS_FAST_INV_SQRT
  reci_norm = q_rsqrt(norm);
#else
  reci_norm = 1.0 / sqrt(norm);
#endif

  a[0] *= reci_norm;
  a[1] *= reci_norm;
  a[2] *= reci_norm;
  a[3] = ang;
}

void quat_to_axang(double *a, const double *q) {
  double q_norm[4];
  normalize_quat(q_norm, q);
  double norm = sqrt(q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]);
  if (norm < __DBL_EPSILON__) {
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    return;
  }
  double reci_norm = 1.0 / norm;
  a[0] = q_norm[1] * reci_norm;
  a[1] = q_norm[2] * reci_norm;
  a[2] = q_norm[3] * reci_norm;
  a[3] = 2.0 * acos(q_norm[0]);
  double a_norm[4];
  normalize_axang(a_norm, a);
  a[0] = a_norm[0];
  a[1] = a_norm[1];
  a[2] = a_norm[2];
  a[3] = a_norm[3];
}

void compact_axang_to_axang(double *a, const double *ca) {
  double ca_norm[3];
  normalize_compact_axang(ca_norm, ca);
  double ang = sqrt(ca_norm[0] * ca_norm[0] + ca_norm[1] * ca_norm[1] + ca_norm[2] * ca_norm[2]);
  if (ang < __DBL_EPSILON__) {
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    return;
  }
  double ang_inv = 1.0 / ang;
  a[0] = ca_norm[0] * ang_inv;
  a[1] = ca_norm[1] * ang_inv;
  a[2] = ca_norm[2] * ang_inv;
  a[3] = ang;
}

void axang_to_compact_axang(double *ca, const double *a) {
  double a_norm[4];
  normalize_axang(a_norm, a);
  ca[0] = a_norm[0] * a_norm[3];
  ca[1] = a_norm[1] * a_norm[3];
  ca[2] = a_norm[2] * a_norm[3];
}

void rmat_to_compact_axang(double *ca, const double *m) {
  double a[4];
  rmat_to_axang(a, m);
  axang_to_compact_axang(ca, a);
}

void quat_to_compact_axang(double *ca, const double *q) {
  double a[4];
  quat_to_axang(a, q);
  axang_to_compact_axang(ca, a);
}

/*
 * Conversions to quaternion
 */
void ang_to_quat(double *q, const int basis, const double ang) {
#if CONVERSIONS_TYPE_CHECK
  assert(basis == 0 || basis == 1 || basis == 2);
#endif
  double half_ang = 0.5 * ang;
  double c = cos(half_ang);
  double s = sin(half_ang);
  q[0] = c;
  q[1] = 0.0;
  q[2] = 0.0;
  q[3] = 0.0;
  q[basis + 1] = s;
}

void rmat_to_quat(double *q, const double *m) {
  assert(is_rmat(m));
  double trace = mat_trace(m, 3);
  double trace_sq;
  if (trace > 0.0) {
    trace_sq = sqrt(1.0 + trace);
    q[0] = 0.5 * trace_sq;
    q[1] = 0.5 / trace_sq * (m[7] - m[5]);
    q[2] = 0.5 / trace_sq * (m[2] - m[6]);
    q[3] = 0.5 / trace_sq * (m[3] - m[1]);
  } else {
    if (m[0] > m[4] && m[0] > m[8]) {
      trace_sq = sqrt(1.0 + m[0] - m[4] - m[8]);
      q[0] = 0.5 / trace_sq * (m[7] - m[5]);
      q[1] = 0.5 * trace_sq;
      q[2] = 0.5 / trace_sq * (m[3] + m[1]);
      q[3] = 0.5 / trace_sq * (m[2] + m[6]);
    } else if (m[4] > m[8]) {
      trace_sq = sqrt(1.0 + m[4] - m[0] - m[8]);
      q[0] = 0.5 / trace_sq * (m[2] - m[6]);
      q[1] = 0.5 / trace_sq * (m[3] + m[1]);
      q[2] = 0.5 * trace_sq;
      q[3] = 0.5 / trace_sq * (m[7] + m[5]);
    } else {
      trace_sq = sqrt(1.0 + m[8] - m[0] - m[4]);
      q[0] = 0.5 / trace_sq * (m[3] - m[1]);
      q[1] = 0.5 / trace_sq * (m[2] + m[6]);
      q[2] = 0.5 / trace_sq * (m[7] + m[5]);
      q[3] = 0.5 * trace_sq;
    }
  }
}

void axang_to_quat(double *q, const double *a) {
  double a_norm[4];
  normalize_axang(a_norm, a);
  double half_theta = a_norm[3] * 0.5;
  double c = cos(half_theta);
  double s = sin(half_theta);
  q[0] = c;
  q[1] = s * a_norm[0];
  q[2] = s * a_norm[1];
  q[3] = s * a_norm[2];
}

void compact_axang_to_quat(double *q, const double *ca) {
  double a[4];
  compact_axang_to_axang(a, ca);
  axang_to_quat(q, a);
}

void euler_to_quat(double *q, const double *e, int i, int j, int k, const bool extrinsic) {
#if CONVERSIONS_TYPE_CHECK
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
#endif
  double alpha = e[0];
  double beta = e[1];
  double gamma = e[2];
  if (!extrinsic) {
    i ^= k;
    k ^= i;
    i ^= k;
    double temp;
    temp = alpha;
    alpha = gamma;
    gamma = temp;
  }
  double q0[4], q1[4], q2[4];
  ang_to_quat(q0, i, alpha);
  ang_to_quat(q1, j, beta);
  ang_to_quat(q2, k, gamma);
  double q21[4];
  quat_product(q21, q2, q1);
  quat_product(q, q21, q0);
}

void quat_xyzw_to_wxyz(double *q_wxyz, const double *q_xyzw) {
  q_wxyz[0] = q_xyzw[3];
  q_wxyz[1] = q_xyzw[0];
  q_wxyz[2] = q_xyzw[1];
  q_wxyz[3] = q_xyzw[2];
}

void quat_wxyz_to_xyzw(double *q_xyzw, const double *q_wxyz) {
  q_xyzw[0] = q_wxyz[1];
  q_xyzw[1] = q_wxyz[2];
  q_xyzw[2] = q_wxyz[3];
  q_xyzw[3] = q_wxyz[0];
}

/*
 * Conversions to Euler angle
 */
void quat_to_euler(double *e, const double *q, int i, int j, int k, const bool extrinsic) {
#if CONVERSIONS_TYPE_CHECK
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
#endif
  double q_norm[4];
  normalize_quat(q_norm, q);
  i++;
  j++;
  k++;

  // The original algorithm assumes extrinsic convention.
  // Hence, we swap the order of axes for intrinsic rotation.
  if (!extrinsic) {
    i ^= k;
    k ^= i;
    i ^= k;
  }

  bool is_proper_euler = (i == k);
  if (is_proper_euler) {
    k = 6 - i - j;
  }

  double sign = (i - j) * (j - k) * (k - i) * 0.5;
  double a = q_norm[0], b = q_norm[i], c = q_norm[j], d = q_norm[k] * sign;
  if (!is_proper_euler) {
    double a_old = a, b_old = b, c_old = c, d_old = d;
    a = a_old - c_old;
    b = b_old + d_old;
    c = c_old + a_old;
    d = d_old - b_old;
  }

  double ang_j = 2.0 * atan2(hypot(c, d), hypot(a, b));
  double eps = 1e-7;
  int singularity;
  if (fabs(ang_j) <= eps) {
    singularity = 1;
  } else if (fabs(ang_j - M_PI) <= eps) {
    singularity = 2;
  } else {
    singularity = 0;
  }

  double half_sum = atan2(b, a);
  double half_diff = atan2(d, c);
  double ang_i, ang_k;
  if (singularity == 0) {
    // no singularity
    ang_i = half_sum + half_diff;
    ang_k = half_sum - half_diff;
  } else if (extrinsic) {
    ang_k = 0.0;
    if (singularity == 1) {
      ang_i = 2.0 * half_sum;
    } else {
      assert(singularity == 2);
      ang_i = 2.0 * half_diff;
    }
  } else {
    ang_i = 0.0;
    if (singularity == 1) {
      ang_k = 2.0 * half_sum;
    } else {
      assert(singularity == 2);
      ang_k = -2.0 * half_diff;
    }
  }

  if (!is_proper_euler) {
    ang_j -= (M_PI * 0.5);
    ang_i *= sign;
  }

  ang_k = normalize_ang(ang_k);
  ang_i = normalize_ang(ang_i);

  if (extrinsic) {
    e[0] = ang_k;
    e[1] = ang_j;
    e[2] = ang_i;
  } else {
    e[0] = ang_i;
    e[1] = ang_j;
    e[2] = ang_k;
  }
}

// ref: https://arc.aiaa.org/doi/abs/10.2514/1.16622
static void active_rmat_to_intrinsic_euler(double *e, const double *m, const double *n1, const double *n2, const double *n3, const bool is_proper_euler) {
  // we followed the naming in the paper
  const double *D = m;
  assert(is_rmat(D));

  // Differences from the paper:
  // 1. we call angles alpha, beta, and gamma
  // 2. we obtain angles from intrinsic rotations

  // eq. 5
  double n1_cross_n2[3];
  cross_product(n1_cross_n2, n1, n2);
  double lambda = atan2(inner_product(n1_cross_n2, n3, 3), inner_product(n1, n3, 3));

  // eq. 6
  double C[3][3];
  int i;
  for (i = 0; i < 3; i++) {
    C[0][i] = n2[i];
    C[1][i] = n1_cross_n2[i];
    C[2][i] = n1[i];
  }

  // eq. 8
  double CT[3][3], CD[3][3], CDCT[3][3], lambda_m[3][3], lambda_mT[3][3], O[3][3];
  mat_multiply(&CD[0][0], &C[0][0], D, 3, 3, 3);
  mat_transpose(&CT[0][0], &C[0][0], 3, 3);
  mat_multiply(&CDCT[0][0], &CD[0][0], &CT[0][0], 3, 3, 3);
  ang_to_rmat(&lambda_m[0][0], 0, lambda, false);
  mat_transpose(&lambda_mT[0][0], &lambda_m[0][0], 3, 3);
  mat_multiply(&O[0][0], &CDCT[0][0], &lambda_mT[0][0], 3, 3, 3);

  double alpha, beta, gamma;
  // fix numerical issue if O_22 is slightly out of range
  double O_22 = fmax(fmin(O[2][2], 1.0), -1.0);
  // eq. 10a
  beta = lambda + acos(O_22);

  double eps = 1e-7;
  bool safe1 = (fabs(beta - lambda) >= eps);
  bool safe2 = (fabs(beta - lambda - M_PI) >= eps);
  if (safe1 && safe2) {
    // default case, no gimbal lock
    // eq. 10b
    alpha = atan2(O[0][2], -O[1][2]);
    // eq. 10c
    gamma = atan2(O[2][0], O[2][1]);

    bool valid_beta;
    if (is_proper_euler) {
      valid_beta = ((0.0 <= beta) && (beta <= M_PI));
    } else {
      // Cardan / Tait-Bryan angles
      double half_pi = 0.5 * M_PI;
      valid_beta = ((-half_pi <= beta) && (beta <= half_pi));
    }
    // eq. 12
    if (!valid_beta) {
      alpha += M_PI;
      beta = 2.0 * lambda - beta;
      gamma -= M_PI;
    }
  } else {
    // gimbal lock
    // a)
    gamma = 0.0;
    if (!safe1) {
      // b)
      alpha = atan2(O[1][0] - O[0][1], O[0][0] + O[1][1]);
    } else {
      alpha = atan2(O[1][0] + O[0][1], O[0][0] - O[1][1]);
    }
  }
  e[0] = normalize_ang(alpha);
  e[1] = normalize_ang(beta);
  e[2] = normalize_ang(gamma);
}

void rmat_to_euler(double *e, const double *m, int i, int j, int k, const bool extrinsic) {
#if CONVERSIONS_TYPE_CHECK
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
#endif
  bool is_proper_euler = (i == k);
  if (extrinsic) {
    i ^= k;
    k ^= i;
    i ^= k;
  }
  double basis_vecs[3][3] = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}
  };
  active_rmat_to_intrinsic_euler(e, m, basis_vecs[i], basis_vecs[j],
                                 basis_vecs[k], is_proper_euler);
  if (extrinsic) {
    double tmp;
    tmp = e[0];
    e[0] = e[2];
    e[2] = tmp;
  }
}