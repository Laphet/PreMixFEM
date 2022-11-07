#ifndef _YCQ_UTIL_H
#define _YCQ_UTIL_H

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#include <math.h>

#define Q0_NUM 1
#define RT0_2D_NUM 4
#define RT0_3D_NUM 6
#define INVALID_INPUT -1
#define UNCLASSIFIED_ERROR -2

extern double RT0_dot_RT0_2d[2];
extern double RT0_dot_RT0_3d[2];
extern double dRT0_Q0_2d;
extern double dRT0_Q0_3d;
extern double Q0_Q0_2d;
extern double Q0_Q0_3d;

typedef struct ModelContext {
  int dim;
  double L, W, H;
  // func kappa;
  // func source;
  // func Diri_BCs[MAX_BC_NUM];
} ModelContext;

/*
extern double elem_stiff_2d[Q0_NUM + RT0_2D_NUM][Q0_NUM + RT0_2D_NUM] = {{0.0}};
extern double elem_stiff_3d[Q0_NUM + RT0_3D_NUM][Q0_NUM + RT0_3D_NUM] = {{0.0}};
extern double elem_divdiv_2d[RT0_2D_NUM][RT0_2D_NUM] = {{0.0}};
extern double elem_divdiv_3d[RT0_3D_NUM][RT0_3D_NUM] = {{0.0}};
*/

// int get_ranks_stencilWidth(int dim, int size, int sub_domain, int M, int N, int P, int *m, int *n, int *p, int *width);

int RT0_2d(double x, double y, int ind, double *phi_x, double *phi_y);
/*
phi_0(x, y) = ((x+1)/2, 0)
phi_1(x, y) = (0, (y+1)/2)
phi_2(x, y) = ((x-1)/2 0)
phi_3(x, y) = (0, (y-1)/2)
*/
int RT0_3d(double x, double y, double z, int ind, double *phi_x, double *phi_y, double *phi_z);
/*
phi_0(x, y, z) = ((x+1)/2, 0, 0)
phi_1(x, y, z) = (0, (y+1)/2, 0)
phi_2(x, y, z) = (0, 0, (z+1)/2)
phi_3(x, y, z) = ((x-1)/2, 0, 0)
phi_4(x, y, z) = (0, (y-1)/2, 0)
phi_5(x, y, z) = (0, 0, (z-1)/2)
*/
int div_RT0_2d(double x, double y, int ind, double *div_phi);
/*
div phi_0(x, y) = div phi_1(x, y)= div phi_2(x, y) = div phi_3(x, y) = 1/4
*/
int div_RT1_2d(double x, double y, double z, int ind, double *div_phi);
/*
div phi_0(x, y) = div phi_1(x, y)= div phi_2(x, y) = div phi_3(x, y) = div phi_4(x, y)= div phi_5(x, y) =1/8
*/
int Q0_2d(double x, double y, double *q0);
/*
q0(x, y) = 1
*/
int Q0_3d(double x, double y, double z, double *q0);
/*
q0(x, y, z) = 1
*/
// int get_elem_mats(void);

double u2d(double x, double y);

double dx_u2d(double x, double y);

double dy_u2d(double x, double y);

double u3d(double x, double y, double z);

double dx_u3d(double x, double y, double z);

double dy_u3d(double x, double y, double z);

double dz_u3d(double x, double y, double z);

double kappa1(double x, double y);

double source1(double x, double y);

double kappa2(double x, double y, double z);

double source2(double x, double y, double z);

double kappa3(double x, double y);

double source3(double x, double y);
// \[Pi]^2 Cos[\[Pi] x] (2 Cos[2 \[Pi] y] Sin[\[Pi] y]+Cos[\[Pi] y] (8+3 Sin[\[Pi] x]+2 Sin[2 \[Pi] y]))

double kappa4(double x, double y, double z);

double source4(double x, double y, double z);
// 1/2 \[Pi]^2 Cos[\[Pi] x] (4 Cos[2 \[Pi] y] Cos[\[Pi] z] Sin[\[Pi] y]+Cos[\[Pi] y] (Cos[\[Pi] z] (24+6 Cos[(\[Pi] z)/2]+8 Sin[\[Pi] x]+6 Sin[2 \[Pi] y])-Sin[(\[Pi] z)/2] Sin[\[Pi] z])){

double source5(double x, double y);

double source6(double x, double y, double z);

#endif