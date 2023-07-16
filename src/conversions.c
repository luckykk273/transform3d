#include "conversions.h"
#include "utils.h"
#include <assert.h>

#define _USE_MATH_DEFINES
#include <math.h>

/*
 * Conversions to rotation matrix
 */
void ang_to_rmat(double (*m)[3], int basis, double ang, bool passive) {
  assert(basis == 0 || basis == 1 || basis == 2);
  double c = cos(ang);
  double s = sin(ang);
  double sign = passive ? 1.0 : -1.0;
  s *= sign;
  if(basis == 0) {
    m[0][0] = 1.0;
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    m[1][0] = 0.0;
    m[1][1] = c;
    m[1][2] = s;
    m[2][0] = 0.0;
    m[2][1] = -s;
    m[2][2] = c;
  } else if(basis == 1) {
    m[0][0] = c;
    m[0][1] = 0.0;
    m[0][2] = -s;
    m[1][0] = 0.0;
    m[1][1] = 1.0;
    m[1][2] = 0.0;
    m[2][0] = s;
    m[2][1] = 0.0;
    m[2][2] = c;
  } else if(basis == 2) {
    m[0][0] = c;
    m[0][1] = s;
    m[0][2] = 0.0;
    m[1][0] = -s;
    m[1][1] = c;
    m[1][2] = 0.0;
    m[2][0] = 0.0;
    m[2][1] = 0.0;
    m[2][2] = 1.0;
  }
}

void euler_to_rmat(double (*m)[3], double *e, int i, int j, int k,
                   bool extrinsic) {
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
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
  ang_to_rmat(m0, i, alpha, false);
  ang_to_rmat(m1, j, beta, false);
  ang_to_rmat(m2, k, gamma, false);
  mat33_dot(m1, m0, m10);
  mat33_dot(m2, m10, m);
}

// void two_vec_to_rmat(double (*m)[3], double *a, double *b) {
//   double norm_a = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
//   assert(norm_a > __DBL_EPSILON__);
//   double norm_b = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
//   assert(norm_b > __DBL_EPSILON__);
//   double c[3];
//   vec_cross(a, b, c);
//   double norm_c = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
//   assert(norm_c > __DBL_EPSILON__);
//   normalize_vec(a);
//   double b_on_a_proj[3], b_on_a_rej[3];
//   vec_proj(b, a, b_on_a_proj);
//   int i;
//   for (i = 0; i < 3; i++) {
//     b_on_a_rej[i] = b[i] - b_on_a_proj[i];
//   }
//   normalize_vec(b_on_a_rej);
//   normalize_vec(c);
//   for (i = 0; i < 3; i++) {
//     m[0][i] = a[i];
//     m[1][i] = b_on_a_rej[i];
//     m[2][i] = c[i];
//   }
// }

void axang_to_rmat(double (*m)[3], double *a) {
  normalize_axang(a);
  double ax = a[0];
  double ay = a[1];
  double az = a[2];
  double theta = a[3];
  double c = cos(theta);
  double s = sin(theta);
  double t = 1.0 - c;
  double tmp1, tmp2;
  m[0][0] = c + ax * ax * t;
  m[1][1] = c + ay * ay * t;
  m[2][2] = c + az * az * t;

  tmp1 = ax * ay * t;
  tmp2 = az * s;
  m[1][0] = tmp1 + tmp2;
  m[0][1] = tmp1 - tmp2;

  tmp1 = ax * az * t;
  tmp2 = ay * s;
  m[2][0] = tmp1 - tmp2;
  m[0][2] = tmp1 + tmp2;

  tmp1 = ay * az * t;
  tmp2 = ax * s;
  m[2][1] = tmp1 + tmp2;
  m[1][2] = tmp1 - tmp2;
}

void compact_axang_to_rmat(double (*m)[3], double *ca) {
  double a[3];
  compact_axang_to_axang(a, ca);
  axang_to_rmat(m, a);
}

void quat_to_rmat(double (*m)[3], double *q) {
  normalize_quat(q);
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double x2 = x + x, y2 = y + y, z2 = z + z;
  double xx = x * x2, xy = x * y2, xz = x * z2;
  double yy = y * y2, yz = y * z2, zz = z * z2;
  double wx = w * x2, wy = w * y2, wz = w * z2;
  m[0][0] = 1.0 - (yy + zz);
  m[0][1] = xy - wz;
  m[0][2] = xz + wy;
  m[1][0] = xy + wz;
  m[1][1] = 1.0 - (xx + zz);
  m[1][2] = yz - wx;
  m[2][0] = xz - wy;
  m[2][1] = yz + wx;
  m[2][2] = 1.0 - (xx + yy);
}

/*
 * Conversions to axis angle
 */
void rmat_to_axang(double *a, double (*m)[3]) {
  assert(is_rmat(m));
  double cos_ang = (mat33_trace(m) - 1.0) / 2.0;
  double ang = acos(fmin(fmax(-1.0, cos_ang), 1.0));
  if (fabs(ang) < __DBL_EPSILON__) {
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    return;
  }
  double ax_unnormalized[3];
  ax_unnormalized[0] = m[2][1] - m[1][2];
  ax_unnormalized[1] = m[0][2] - m[2][0];
  ax_unnormalized[2] = m[1][0] - m[0][1];
  if (fabs(ang - M_PI) < 1e-4) {
    double m_diag[3], eeT_diag[3];
    double scale;
    int i;
    for (i = 0; i < 3; i++) {
      m_diag[i] = clip(m[i][i], -1.0, 1.0);
      eeT_diag[i] = 0.5 * (m_diag[i] + 1.0);
      scale = (sign(ax_unnormalized[i]) == -1) ? -1.0 : 1.0;
      a[i] = sqrt(eeT_diag[i]) * scale;
    }
  } else {
    a[0] = ax_unnormalized[0];
    a[1] = ax_unnormalized[1];
    a[2] = ax_unnormalized[2];
  }
  double norm = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  a[0] /= norm;
  a[1] /= norm;
  a[2] /= norm;
  a[3] = ang;
}

void quat_to_axang(double *a, double *q) {
  normalize_quat(q);
  double norm = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm < __DBL_EPSILON__) {
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    return;
  }
  a[0] = q[1] / norm;
  a[1] = q[2] / norm;
  a[2] = q[3] / norm;
  a[3] = 2.0 * acos(q[0]);
  normalize_axang(a);
}

void compact_axang_to_axang(double *a, double *ca) {
  normalize_compact_axang(ca);
  double ang = sqrt(ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2]);
  if (ang < __DBL_EPSILON__) {
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    return;
  }
  a[0] = ca[0] / ang;
  a[1] = ca[1] / ang;
  a[2] = ca[2] / ang;
  a[3] = ang;
}

void axang_to_compact_axang(double *ca, double *a) {
  normalize_axang(a);
  ca[0] = a[0] * a[3];
  ca[1] = a[1] * a[3];
  ca[2] = a[2] * a[3];
}

void rmat_to_compact_axang(double *ca, double (*m)[3]) {
  double a[4];
  rmat_to_axang(a, m);
  axang_to_compact_axang(ca, a);
}

void quat_to_compact_axang(double *ca, double *q) {
  double a[4];
  quat_to_axang(a, q);
  axang_to_compact_axang(ca, a);
}

/*
 * Conversions to quaternion
 */
void ang_to_quat(double *q, int basis, double ang) {
  assert(basis == 0 || basis == 1 || basis == 2);
  double half_ang = 0.5 * ang;
  double c = cos(half_ang);
  double s = sin(half_ang);
  q[0] = c;
  q[1] = 0.0;
  q[2] = 0.0;
  q[3] = 0.0;
  q[basis + 1] = s;
}

void rmat_to_quat(double *q, double (*m)[3]) {
  assert(is_rmat(m));
  double trace = mat33_trace(m);
  double trace_sq;
  if (trace > 0.0) {
    trace_sq = sqrt(1.0 + trace);
    q[0] = 0.5 * trace_sq;
    q[1] = 0.5 / trace_sq * (m[2][1] - m[1][2]);
    q[2] = 0.5 / trace_sq * (m[0][2] - m[2][0]);
    q[3] = 0.5 / trace_sq * (m[1][0] - m[0][1]);
  } else {
    if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
      trace_sq = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
      q[0] = 0.5 / trace_sq * (m[2][1] - m[1][2]);
      q[1] = 0.5 * trace_sq;
      q[2] = 0.5 / trace_sq * (m[1][0] + m[0][1]);
      q[3] = 0.5 / trace_sq * (m[0][2] + m[2][0]);
    } else if (m[1][1] > m[2][2]) {
      trace_sq = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
      q[0] = 0.5 / trace_sq * (m[0][2] - m[2][0]);
      q[1] = 0.5 / trace_sq * (m[1][0] + m[0][1]);
      q[2] = 0.5 * trace_sq;
      q[3] = 0.5 / trace_sq * (m[2][1] + m[1][2]);
    } else {
      trace_sq = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
      q[0] = 0.5 / trace_sq * (m[1][0] - m[0][1]);
      q[1] = 0.5 / trace_sq * (m[0][2] + m[2][0]);
      q[2] = 0.5 / trace_sq * (m[2][1] + m[1][2]);
      q[3] = 0.5 * trace_sq;
    }
  }
}

void axang_to_quat(double *q, double *a) {
  normalize_axang(a);
  double half_theta = a[3] * 0.5;
  double c = cos(half_theta);
  double s = sin(half_theta);
  q[0] = c;
  q[1] = s * a[0];
  q[2] = s * a[1];
  q[3] = s * a[2];
}

void compact_axang_to_quat(double *q, double *ca) {
  double a[4];
  compact_axang_to_axang(a, ca);
  axang_to_quat(q, a);
}

void euler_to_quat(double *q, double *e, int i, int j, int k, bool extrinsic) {
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
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
  quat_mul(q2, q1, q21);
  quat_mul(q21, q0, q);
}

// void mrp_to_quat(double *q, double *mrp) {
//   double dot_p1 = vec_dot(mrp, mrp) + 1.0;
//   q[0] = (2.0 - dot_p1) / dot_p1;
//   int i;
//   for (i = 1; i < 4; i++) {
//     q[i] = 2.0 * mrp[i] / dot_p1;
//   }
// }

void quat_xyzw_to_wxyz(double *q) {
  double qw = q[3];
  int i;
  for (i = 2; i > -1; i--) {
    q[i + 1] = q[i];
  }
  q[0] = qw;
}

void quat_wxyz_to_xyzw(double *q) {
  double qw = q[0];
  int i;
  for (i = 1; i < 4; i++) {
    q[i - 1] = q[i];
  }
  q[3] = qw;
}

/*
 * Conversions to Euler angle
 */
void quat_to_euler(double *e, double *q, int i, int j, int k, bool extrinsic) {
  normalize_quat(q);
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
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
  double a = q[0], b = q[i], c = q[j], d = q[k] * sign;
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
static void active_rmat_to_intrinsic_euler(double *e, double (*m)[3], double *n1,
                                           double *n2, double *n3,
                                           bool is_proper_euler) {
  // we followed the naming in the paper
  double(*D)[3] = m;
  assert(is_rmat(D));

  // Differences from the paper:
  // 1. we call angles alpha, beta, and gamma
  // 2. we obtain angles from intrinsic rotations

  // eq. 5
  double n1_cross_n2[3];
  vec_cross(n1, n2, n1_cross_n2);
  double lambda = atan2(vec_dot(n1_cross_n2, n3), vec_dot(n1, n3));

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
  mat33_dot(C, D, CD);
  mat33_transpose(C, CT);
  mat33_dot(CD, CT, CDCT);
  ang_to_rmat(lambda_m, 0, lambda, false);
  mat33_transpose(lambda_m, lambda_mT);
  mat33_dot(CDCT, lambda_mT, O);

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

void rmat_to_euler(double *e, double (*m)[3], int i, int j, int k,
                   bool extrinsic) {
  assert(i == 0 || i == 1 || i == 2);
  assert(j == 0 || j == 1 || j == 2);
  assert(k == 0 || k == 1 || k == 2);
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