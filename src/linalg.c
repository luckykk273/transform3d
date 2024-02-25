#include "matrix.h"
#include "vector.h"
#include "linalg.h"
#include "trans_utils.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <stdbool.h>
#include <string.h>
#if LINALG_TYPE_CHECK
#include <assert.h>
#endif

size_t lu_factorize(double *m, const size_t r, const size_t c) {
#if LINALG_TYPE_CHECK
  double cm[MAT_MAX_SIZE][MAT_MAX_SIZE];
  memcpy(cm, m, r * c * sizeof(double));
#endif
  size_t singular = 0;
  size_t size = (r < c) ? r : c;
  size_t i, j;
  double outer_prod[MAT_MAX_SIZE][MAT_MAX_SIZE] = {0};
  double mci[MAT_MAX_SIZE], mri[MAT_MAX_SIZE];
  for(i = 0; i < size; ++i) {
    // if(m[i * c + i] != 0.0) {
    if(!is_bit_zero(m[i * c + i])) {
      double m_inv = 1.0 / m[i * c + i];
      for(j = i + 1; j < r; ++j) {
        m[j * c + i] *= m_inv;
      }
    } else if (singular == 0) {
      singular = i + 1;
    }
    
    mat_column(mci, m, r, c, i);
    mat_row(mri, m, r, c, i);
    outer_product(&outer_prod[0][0], &mci[i + 1], &mri[i + 1], r - (i + 1), c - (i + 1));
    mat_subtract(&m[(i + 1) * (c + 1)], &m[(i + 1) * (c + 1)], &outer_prod[0][0], c, c, c - (i + 1), r - (i + 1), c - (i + 1));
  } 
#if LINALG_TYPE_CHECK
  assert(singular == 0);
  double m_mul[MAT_MAX_SIZE][MAT_MAX_SIZE], upper_triangular[MAT_MAX_SIZE][MAT_MAX_SIZE], lower_triangular[MAT_MAX_SIZE][MAT_MAX_SIZE];
  mat_upper(&upper_triangular[0][0], m, r, c, false);
  mat_lower(&lower_triangular[0][0], m, r, c, true);
  mat_multiply(&m_mul[0][0], &lower_triangular[0][0], &upper_triangular[0][0], r, c, c);
  for(i = 0; i < r; ++i) {
    for(j = 0; j < c; ++j) {
      assert(fabs(m_mul[i][j] - cm[i][j]) < 1e-8);
    }
  }
#endif
  return singular;
}

// LU factorization with partial pivoting
size_t lu_factorize_pivot(double *m, size_t *pm, const size_t r, const size_t c) {
#if LINALG_TYPE_CHECK
  double cm[MAT_MAX_SIZE][MAT_MAX_SIZE];
  memcpy(cm, m, r * c * sizeof(double));
#endif
  size_t singular = 0;
  size_t size = (r < c) ? r : c;
  size_t i, j, i_norm_inf;
  double outer_prod[MAT_MAX_SIZE][MAT_MAX_SIZE] = {0};
  double mci[MAT_MAX_SIZE], mri[MAT_MAX_SIZE];
  for(i = 0; i < size; ++i) {
    mat_column(mci, m, r, c, i);
    i_norm_inf = i + index_norm_inf(&mci[i], r - i);
#if LINALG_TYPE_CHECK
    assert(i_norm_inf < r);
#endif
    // if(m[i_norm_inf * c + i] != 0.0) {
    if(!is_bit_zero(m[i_norm_inf * c + i])) {
      if(i_norm_inf != i) {
        pm[i] = i_norm_inf;
        mat_swap_rows(m, r, c, i_norm_inf, i);
      } else {
#if LINALG_TYPE_CHECK
        assert(pm[i] == i_norm_inf);
#endif
      }
      double m_inv = 1.0 / m[i * c + i];
      for(j = i + 1; j < r; ++j) {
        m[j * c + i] *= m_inv;
      }
    } else if (singular == 0) {
      singular = i + 1;
    }

    mat_column(mci, m, r, c, i);
    mat_row(mri, m, r, c, i);
    outer_product(&outer_prod[0][0], &mci[i + 1], &mri[i + 1], r - (i + 1), c - (i + 1));
    mat_subtract(&m[(i + 1) * (c + 1)], &m[(i + 1) * (c + 1)], &outer_prod[0][0], c, c, c - (i + 1), r - (i + 1), c - (i + 1));
  } 
#if LINALG_TYPE_CHECK
  assert(singular == 0);
  for(i = 0; i < r; ++i) {
    if(i != pm[i]) {
      mat_swap_rows(&cm[0][0], r, c, i, pm[i]);
    }
  }
  
  double m_mul[MAT_MAX_SIZE][MAT_MAX_SIZE], upper_triangular[MAT_MAX_SIZE][MAT_MAX_SIZE], lower_triangular[MAT_MAX_SIZE][MAT_MAX_SIZE];
  mat_upper(&upper_triangular[0][0], m, r, c, false);
  mat_lower(&lower_triangular[0][0], m, r, c, true);
  mat_multiply(&m_mul[0][0], &lower_triangular[0][0], &upper_triangular[0][0], r, c, c);
  for(i = 0; i < r; ++i) {
    for(j = 0; j < c; ++j) {
      assert(fabs(m_mul[i][j] - cm[i][j]) < 1e-8);
    }
  }
#endif
  return singular;
}

static int determinant_sign(const size_t *pm, const size_t n) {
  int pm_sign = 1;
  size_t i;
  for(i = 0; i < n; ++i) {
    if(i != pm[i]) {
      pm_sign *= -1;
    }
  }
  return pm_sign;
}


// Note: Here we expand all matrix multiply with size <= 3.
// If one wants to change to size <= 5, the following reference can be refered:
// https://www.geeksforgeeks.org/cpp-program-for-determinant-of-a-matrix/
static double determinant_small(const double *m, const size_t n) {
#if LINALG_TYPE_CHECK
  assert(n <= 3);
#endif
  if(n == 1) {
    return m[0];
  } else if(n == 2) {
    return m[0] * m[3] - m[1] * m[2];
  } else if(n == 3) {
    return (
      m[0] * m[4] * m[8] + 
      m[1] * m[5] * m[6] + 
      m[2] * m[3] * m[7] -
      m[0] * m[5] * m[7] -
      m[1] * m[3] * m[8] -
      m[2] * m[4] * m[6]
    );
  }
  return 0.0;
}

double determinant(double *m, const size_t n) {
#if LINALG_TYPE_CHECK
  assert(n <= MAT_MAX_SIZE);
#endif
  if(n <= 3) {
    return determinant_small(m, n);
  }
  
  size_t pm[MAT_MAX_SIZE];
  size_t i;
  for(i = 0; i < n; ++i) {
    pm[i] = i;
  }

  double det = 1.0;
  if(lu_factorize_pivot(m, pm, n, n)) {
    det = 0.0;
  } else {
    for(i = 0; i < n; ++i) {
      det *= m[i * n + i];
    }
    det *= determinant_sign(pm, n);
  }

  return det;
}