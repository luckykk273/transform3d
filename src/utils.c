#include "utils.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

/*
 * Helper functions
 */
double determinant(double (*m)[3]) {
  double det = m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +
               m[0][2] * m[1][0] * m[2][1] - m[0][0] * m[1][2] * m[2][1] -
               m[0][1] * m[1][0] * m[2][2] - m[0][2] * m[1][1] * m[2][0];
  return det;
}

double clip(double x, double min, double max) {
  // return (x < min) ? min : (x > max ? max : x);
  if (x < min) {
    return min;
  }
  if (x > max) {
    return max;
  }
  return x;
}

int sign(double x) {
  if (x < 0.0) {
    return -1;
  }
  if (x > 0.0) {
    return 1;
  }
  return 0;
}

double mat33_trace(double (*m)[3]) { return (m[0][0] + m[1][1] + m[2][2]); }

void mat33_dot(double (*m1)[3], double (*m2)[3], double (*m12)[3]) {
  int i, j, k;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      m12[i][j] = 0.0;
      for (k = 0; k < 3; k++) {
        m12[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
}

void mat33_transpose(double (*m)[3], double (*mT)[3]) {
  int i, j;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      mT[i][j] = m[j][i];
    }
  }
}

double vec_dot(double *a, double *b) {
  double res = 0.0;
  int i;
  for (i = 0; i < 3; i++) {
    res += a[i] * b[i];
  }
  return res;
}

void vec_cross(double *v1, double *v2, double *v_res) {
  v_res[0] = v1[1] * v2[2] - v1[2] * v2[1];
  v_res[1] = v1[2] * v2[0] - v1[0] * v2[2];
  v_res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void vec_proj(double *a, double *b, double *a_on_b) {
  double eps = 1e-6;
  double b_sq = vec_dot(b, b);
  if (b_sq < eps) {
    a_on_b[0] = 0.0;
    a_on_b[1] = 0.0;
    a_on_b[2] = 0.0;
    return;
  }

  double a_dot_b = vec_dot(a, b);
  double scale = a_dot_b / b_sq;
  int i;
  for (i = 0; i < 3; i++) {
    a_on_b[i] = b[i] * scale;
  }
}

void quat_prod_vec(double *q, double *v, double *v_res) {
  normalize_quat(q);
  double t[3], cross_qt[3];
  vec_cross(&q[1], v, t);
  int i;
  for (i = 0; i < 3; i++) {
    t[i] *= 2.0;
  }
  vec_cross(&q[1], t, cross_qt);
  for (i = 0; i < 3; i++) {
    v_res[i] = v[i] + q[0] * t[i] + cross_qt[i];
  }
}

void quat_mul(double *q0, double *q1, double *q01) {
  // normalize_quat(q0);
  // normalize_quat(q1);
  q01[0] = q0[0] * q1[0] - vec_dot(&q0[1], &q1[1]);
  double q01_cross[3];
  vec_cross(&q0[1], &q1[1], q01_cross);
  int i;
  for (i = 1; i < 4; i++) {
    q01[i] = q0[0] * q1[i] + q1[0] * q0[i] + q01_cross[i-1];
  }
}

void quat_conj(double *q, double *q_conj) {
  normalize_quat(q);
  q_conj[0] = q[0];
  q_conj[1] = -q[1];
  q_conj[2] = -q[2];
  q_conj[3] = -q[3];
}

/*
 * Validation functions
 */
bool is_rmat(double (*m)[3]) {
  double eps = 1e-6;
  if (fabs(m[0][0] * m[0][1] + m[1][0] * m[1][1] + m[2][0] * m[2][1]) > eps)
    return false;
  if (fabs(m[0][0] * m[0][2] + m[1][0] * m[1][2] + m[2][0] * m[2][2]) > eps)
    return false;
  if (fabs(m[0][1] * m[0][2] + m[1][1] * m[1][2] + m[2][1] * m[2][2]) > eps)
    return false;
  if (fabs(m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0] - 1.0) > eps)
    return false;
  if (fabs(m[0][1] * m[0][1] + m[1][1] * m[1][1] + m[2][1] * m[2][1] - 1.0) > eps)
    return false;
  if (fabs(m[0][2] * m[0][2] + m[1][2] * m[1][2] + m[2][2] * m[2][2] - 1.0) > eps)
    return false;
  return (fabs(determinant(m) - 1.0) < eps);
}


double normalize_ang(double ang) {
  double TWO_PI = 2.0 * M_PI;
  return (ang - (ceil((ang + M_PI) / TWO_PI) - 1.0) * TWO_PI);  // (-Pi, Pi]
  // return (ang - floor((ang + M_PI) / TWO_PI) * TWO_PI);  // [-Pi, Pi)
}


void normalize_vec(double *v) {
  double eps = 1e-6;
  double norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (norm < eps) {
    return;
  }
  v[0] /= norm;
  v[1] /= norm;
  v[2] /= norm;
}

void normalize_axang(double *a) {
  double ang = a[3];
  double norm = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  if (fabs(ang) < __DBL_EPSILON__ || norm < __DBL_EPSILON__) {
    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = 0.0;
    return;
  }
  a[0] /= norm;
  a[1] /= norm;
  a[2] /= norm;
  ang = normalize_ang(ang);
  if (ang < 0.0) {
    ang = -ang;
    a[0] = -a[0];
    a[1] = -a[1];
    a[2] = -a[2];
  }
  a[3] = ang;
}

void normalize_compact_axang(double *ca) {
  double eps = 1e-6;
  double ang = sqrt(ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2]);
  if (ang < eps) {
    ca[0] = 0.0;
    ca[1] = 0.0;
    ca[2] = 0.0;
    return;
  }
  ca[0] /= ang;
  ca[1] /= ang;
  ca[2] /= ang;
  ang = normalize_ang(ang);
  ca[0] *= ang;
  ca[1] *= ang;
  ca[2] *= ang;
}

void normalize_quat(double *q) {
  double eps = 1e-6;
  double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm < eps) {
    return;
  }
  q[0] /= norm;
  q[1] /= norm;
  q[2] /= norm;
  q[3] /= norm;
}
