#include "conversions.h"
#include "utils.h"
#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>

int main(void) {
  double a[4] = {-0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 5e-08};
  double m[3][3];
  double a2[4];
  axang_to_rmat(m, a);
  rmat_to_axang(a2, m);

  return 0;
}