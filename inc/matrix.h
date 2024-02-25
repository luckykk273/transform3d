#ifndef TRANS_MATRIX_H_
#define TRANS_MATRIX_H_

#include <stdint.h>
#include <stdbool.h>

#define MAT_TYPE_CHECK  (1U)
#define MAT_MAX_SIZE    (12U)

#ifdef __cplusplus
extern "C" {
#endif

void mat_row(double *mr, const double *m, const size_t r, const size_t c, const size_t n);

void mat_column(double *mc, const double *m, const size_t r, const size_t c, const size_t n);

void mat_swap_rows(double *m, const size_t r, const size_t c, const size_t r1, const size_t r2);

void mat_swap_columns(double *m, const size_t r, const size_t c, const size_t c1, const size_t c2);

// m_sub, m1, m2 should pass the start position;
// c_sub, c1, c2 should pass the column of the whole matrix;
// lr, lc should pass the length of rows and columns you want to subtract.
void mat_subtract(double *m_sub, const double *m1, const double *m2, const size_t c_sub, const size_t c1, const size_t c2, const size_t lr, const size_t lc);

void mat_add(double *m_add, const double *m1, const double *m2, const size_t c_add, const size_t c1, const size_t c2, const size_t lr, const size_t lc);

void mat_mul_element_wise(double *m_mul, const double *m1, const double *m2, const size_t c_mul, const size_t c1, const size_t c2, const size_t lr, const size_t lc);

void mat_multiply(double *m12, const double *m1, const double *m2, const size_t r1, const size_t c1, const size_t c2);

void mat_mul_scalar(double *m_mul, const double *m, const size_t r, const size_t c, const double scalar);

void mat_upper(double *upper, const double *m, const size_t r, const size_t c, const bool is_unit);

void mat_lower(double *lower, const double *m, const size_t r, const size_t c, const bool is_unit);

// The trace is only defined for a square matrix n×n.
double mat_trace(const double *m, const size_t n);

void mat_transpose(double *mt, const double *m, const size_t r, const size_t c);

// Set all elements in the matrix to n
void mat_all_n(double *m, const size_t r, const size_t c, const double n);

// Set the elements on the diagonal in the matrix to n; o.w. zeros
void mat_scalar(double *m, const size_t r, const size_t c, const double scalar);

#ifdef __cplusplus
}
#endif

#endif  // TRANS_MATRIX_H_