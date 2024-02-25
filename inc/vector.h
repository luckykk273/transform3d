#ifndef TRANS_VECTOR_H_
#define TRANS_VECTOR_H_

#include <stdint.h>
#include <stdbool.h>

#define VEC_TYPE_CHECK  (1U)
#define VEC_MAX_SIZE    (12U)

#ifdef __cplusplus
extern "C" {
#endif

void outer_product(double *outer, const double *u, const double *v, const size_t m, const size_t n);

double inner_product(const double *u, const double *v, const size_t n);

double norm_inf(double *v, size_t n);

// Note: return the minimum index if 2 or more elements are equal.
size_t index_norm_inf(double *v, size_t n);

double vec_magnitude(const double *v, size_t n);

// Note: The cross product is an operation on two vectors in a 3D Euclidean space.
void cross_product(double *cross, const double *u, const double *v);

void orthogonal_project(double *proj, const double *u, const double *v, const size_t n);

void normalize_vector(double *v_norm, const double *v, const size_t n);

double angle_between_vectors(const double *u, const double *v, const bool fast, const size_t n);

// lerp = (1 - t) * u + t * v = u - t * (u - v)
void vec_lerp(double *lerp, const double *u, const double *v, const double t, const size_t n);

void vec_slerp(double *slerp, const double *u, const double *v, const double t, const size_t n);

#ifdef __cplusplus
}
#endif

#endif  // TRANS_VECTOR_H_