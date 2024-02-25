#include "vector.h"
#include "trans_utils.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <stdbool.h>
#if VEC_TYPE_CHECK
#include <assert.h>
#endif



void outer_product(double *outer, const double *u, const double *v, const size_t m, const size_t n) {
  size_t i, j;
  for(i = 0; i < m; ++i) {
    for(j = 0; j < n; ++j) {
      outer[i * n + j] = u[i] * v[j];
    }
  }
}

double inner_product(const double *u, const double *v, const size_t n) {
  size_t i;
  double scalar = 0.0;
  for(i = 0; i < n; ++i) {
    scalar += u[i] * v[i];
  }
  return scalar;
}

double norm_inf(double *v, size_t n) {
  size_t i;
  double max = -1.0;
  for(i = 0; i < n; ++i) {
    if(fabs(v[i]) > max) {
      max = fabs(v[i]);
    }
  }
  return max;
}

size_t index_norm_inf(double *v, size_t n) {
  size_t index;
  double max = -1.0;
  while(n) {
    n--;
    if(fabs(v[n]) > max) {
      max = fabs(v[n]);
      index = n;
    }
  }
  return index;
}

double vec_magnitude(const double *v, size_t n) {
  return sqrt(inner_product(v, v, n));
}

void cross_product(double *cross, const double *u, const double *v) {
  cross[0] = u[1] * v[2] - u[2] * v[1];
  cross[1] = u[2] * v[0] - u[0] * v[2];
  cross[2] = u[0] * v[1] - u[1] * v[0];
}

void orthogonal_project(double *proj, const double *u, const double *v, const size_t n) {
  size_t i;
  double eps = 1e-6;
  double v_sq = inner_product(v, v, n);
  if(v_sq < eps) {
    for(i = 0; i < n; ++i) {
      proj[i] = 0.0;
    }
    return;
  }

  double u_dot_v = inner_product(u, v, n);
  double scalar = u_dot_v / v_sq;
  for(i = 0; i < n; ++i) {
    proj[i] = v[i] * scalar;
  }
}

void normalize_vector(double *v_norm, const double *v, const size_t n) {
  double eps = 1e-12, norm = 0.0;
  size_t i;
  for(i = 0; i < n; ++i) {
    norm += v[i] * v[i];
  }

  if (norm < eps) {
    return;
  }
  
  double reci_norm;
#if UTILS_FAST_INV_SQRT
  reci_norm = q_rsqrt(norm);
#else
  reci_norm = 1.0 / sqrt(norm);
#endif
  for(i = 0; i < n; ++i) {
    v_norm[i] = v[i] * reci_norm;
  }
}

double angle_between_vectors(const double *u, const double *v, const bool fast, const size_t n) {
  double u_dot_v = inner_product(u, v, n);
  if(n != 3 || fast) {
    double norm_u = inner_product(u, u, n);
    double norm_v = inner_product(v, v, n);
    double norm_u_inv, norm_v_inv;
#if UTILS_FAST_INV_SQRT
    norm_u_inv = q_rsqrt(norm_u);
    norm_v_inv = q_rsqrt(norm_v);
#else
    norm_u_inv = 1.0 / sqrt(norm_u);
    norm_v_inv = 1.0 / sqrt(norm_v);
#endif
    return acos(clip(u_dot_v * norm_u_inv * norm_v_inv, -1.0, 1.0));
  }

  double u_cross_v[3];
  cross_product(u_cross_v, u, v);
  return atan2(sqrt(inner_product(u_cross_v, u_cross_v, 3)), u_dot_v);
}

void vec_lerp(double *lerp, const double *u, const double *v, const double t, const size_t n) {
  size_t i;
  for(i = 0; i < n; ++i) {
    // lerp[i] = (1.0 - t) * u[i] + t * v[i];
    lerp[i] = u[i] - t * (u[i] - v[i]);
  }
}

static void vec_slerp_weights(double *w1, double *w2, const double angle, const double t) {
  // if(angle == 0.0) {
  if(is_bit_zero(angle)) {
    *w1 = 1.0;
    *w2 = 0.0;
    return;
  }
  double s_inv = 1.0 / sin(angle);
  *w1 = sin((1.0 - t) * angle) * s_inv;
  *w2 = sin(t * angle) * s_inv;
}

void vec_slerp(double *slerp, const double *u, const double *v, const double t, const size_t n) {
  double angle = angle_between_vectors(u, v, false, n);
  double w1, w2;
  vec_slerp_weights(&w1, &w2, angle, t);
  size_t i;
  for(i = 0; i < n; ++i) {
    slerp[i] = w1 * u[i] + w2 * v[i];
  }
}

