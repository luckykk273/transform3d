#include "matrix.h"
#include "trans_utils.h"

#define _USE_MATH_DEFINES
#include <math.h>


#include <stdbool.h>
#if MAT_TYPE_CHECK
#include <assert.h>
#endif


void mat_row(double *mr, const double *m, const size_t r, const size_t c, const size_t n) {
#if MAT_TYPE_CHECK
  assert(n < r);
#endif
  size_t i;
  for(i = 0; i < c; ++i) {
    mr[i] = m[n * c + i];
  }
}

void mat_column(double *mc, const double *m, const size_t r, const size_t c, const size_t n) {
#if MAT_TYPE_CHECK
  assert(n < c);
#endif
  size_t i;
  for (i = 0; i < r; ++i) {
    mc[i] = m[i * c + n];
  }
}

void mat_swap_rows(double *m, const size_t r, const size_t c, const size_t r1, const size_t r2) {
#if MAT_TYPE_CHECK
  assert(r1 < r && r2 < r);
#endif
  size_t i;
  for(i = 0; i < c; ++i) {
    swap(&m[r1 * c + i], &m[r2 * c + i]);
  }
}

void mat_swap_columns(double *m, const size_t r, const size_t c, const size_t c1, const size_t c2) {
#if MAT_TYPE_CHECK
  assert(c1 < c && c2 < c);
#endif
  size_t i;
  for(i = 0; i < r; ++i) {
    swap(&m[i * r + c1], &m[i * r + c2]);
  }
}

void mat_subtract(double *m_sub, const double *m1, const double *m2, const size_t c_sub, const size_t c1, const size_t c2, const size_t lr, const size_t lc) {
  size_t i, j;
  for(i = 0; i < lr; ++i) {
    for(j = 0; j < lc; ++j) {
      m_sub[i * c_sub + j] = m1[i * c1 + j] - m2[i * c2 + j];
    }
  }
}

void mat_add(double *m_add, const double *m1, const double *m2, const size_t c_add, const size_t c1, const size_t c2, const size_t lr, const size_t lc) {
  size_t i, j;
  for(i = 0; i < lr; ++i) {
    for(j = 0; j < lc; ++j) {
      m_add[i * c_add + j] = m1[i * c1 + j] + m2[i * c2 + j];
    }
  }
}

void mat_mul_element_wise(double *m_mul, const double *m1, const double *m2, const size_t c_mul, const size_t c1, const size_t c2, const size_t lr, const size_t lc) {
  size_t i, j;
  for(i = 0; i < lr; ++i) {
    for(j = 0; j < lc; ++j) {
      m_mul[i * c_mul + j] = m1[i * c1 + j] * m2[i * c2 + j];
    }
  }
}

void mat_multiply(double *m12, const double *m1, const double *m2, const size_t r1, const size_t c1, const size_t c2) {
  size_t i, j, k;
  double sum;
  for(i = 0; i < r1; ++i) {
    for(j = 0; j < c2; ++j) {
      sum = 0.0;
      for(k = 0; k < c1; ++k) {
        sum += (m1[i * c1 + k] * m2[k * c2 + j]);
      }
      m12[i * c2 + j] = sum;
    }
  }
}

void mat_mul_scalar(double *m_mul, const double *m, const size_t r, const size_t c, const double scalar) {
  size_t i, j;
  for(i = 0; i < r; ++i) {
    for(j = 0; j < c; ++j) {
      m_mul[i * c + j] = m[i * c + j] * scalar;
    }
  }
}

void mat_upper(double *upper, const double *m, const size_t r, const size_t c, const bool is_unit) {
  size_t i, j;
  for(i = 0; i < r; ++i) {
    for(j = 0; j < c; ++j) {
      if(j >= i) {
        upper[i * c + j] = (is_unit && i == j) ? 1.0 : m[i * c + j];
      } else {
        upper[i * c + j] = 0.0;
      }
    }
  }
}

void mat_lower(double *lower, const double *m, const size_t r, const size_t c, const bool is_unit) {
  size_t i, j;
  for(i = 0; i < r; ++i) {
    for(j = 0; j < c; ++j) {
      if(j <= i) {
        lower[i * c + j] = (is_unit && i == j) ? 1.0 : m[i * c + j];
      } else {
        lower[i * c + j] = 0.0;
      }
    }
  }
}

double mat_trace(const double *m, const size_t n) {
  size_t i;
  double trace = 0.0;
  for(i = 0; i < n; ++i) {
    trace += m[i * n + i];
  }
  return trace;
}

void mat_transpose(double *mt, const double *m, const size_t r, const size_t c) {
  size_t i, j;
  for(i = 0; i < r; ++i) {
    for(j = 0; j < c; ++j) {
      mt[i * c + j] = m[j * c + i];
    }
  }
}

void mat_all_n(double *m, const size_t r, const size_t c, const double n) {
  size_t i;
  for(i = 0; i < r * c; ++i) {
    m[i] = n;
  }
}

void mat_scalar(double *m, const size_t r, const size_t c, const double scalar) {
  size_t i, j;
  for(i = 0; i < r; ++i) {
    for(j = 0; j < c; ++j) {
      m[i * c + j] = (i == j) ? scalar : 0.0;
    }
  }
}