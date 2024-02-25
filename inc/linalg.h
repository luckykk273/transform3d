#ifndef TRANS_LINALG_H_
#define TRANS_LINALG_H_

#include <stdint.h>

#define LINALG_TYPE_CHECK  (1U)

#ifdef __cplusplus
extern "C" {
#endif


// LU factorization without pivoting
// Note: m will be modified
size_t lu_factorize(double *m, const size_t r, const size_t c);

// LU factorization with partial pivoting
// Note: m, pm will be modified
size_t lu_factorize_pivot(double *m, size_t *pm, const size_t r, const size_t c);

// Desc.:
// Use LU decomposition to calculate the determinant as boost:
// M = LU, where L and U are lower and upper triangular matrices;
// Because the determinant of the triangular matrix is the product of the diagnal elements:
// det(L) = L_d1 * L_d2 * ... * L_dn
// det(U) = U_d1 * U_d2 * ... * U_dn
// By Cauchy–Binet formula, det(AB) = det(A) * det(B), det(M) = det(L) * det(U).
// Because L is the matrix which diagnal elements are all 1,
// det(M) = det(U)
// 
// // Note: If a `small` matrix(3x3 here, but maybe 5x5),
//       A brute-force method is much faster than using LU decomposition.
//
// ref: 
// 1. https://www.boost.org/doc/libs/1_52_0/boost/numeric/ublas/lu.hpp
// 2. http://programmingexamples.net/wiki/CPP/Boost/Math/uBLAS/determinant
double determinant(double *m, const size_t n);


#ifdef __cplusplus
}
#endif

#endif  // TRANS_LINALG_H_