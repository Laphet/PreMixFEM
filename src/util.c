#include "util.h"

double RT0_dot_RT0_2d[2] = {4. / 3., 2. / 3.};
double RT0_dot_RT0_3d[2] = {8. / 3., 4. / 3.};
double dRT0_Q0_2d = 2.0;
double dRT0_Q0_3d = 4.0;
double Q0_Q0_2d = 4.0;
double Q0_Q0_3d = 8.0;

// int get_ranks_stencilWidth(int dim, int size, int sub_domain, int M, int N, int P, int *re_m, int *re_n, int *re_p, int *re_width)
// {
//     int m, n, p, pm, m_r, n_r, p_r, max_r, width;
//     switch (dim)
//     {
//     case 3:
//         n = (int)(0.5 + pow(((double)N * N) * ((double)size) / ((double)P * M), (double)(1. / 3.)));
//         if (!n)
//             n = 1;
//         while (n > 0)
//         {
//             pm = size / n;
//             if (n * pm == size)
//                 break;
//             n--;
//         }
//         if (!n)
//             n = 1;
//         m = (int)(0.5 + sqrt(((double)M) * ((double)size) / ((double)P * n)));
//         if (!m)
//             m = 1;
//         while (m > 0)
//         {
//             p = size / (m * n);
//             if (m * n * p == size)
//                 break;
//             m--;
//         }
//         if (M > P && m < p)
//         {
//             int _m = m;
//             m = p;
//             p = _m;
//         }
//         if (m * n * p != size)
//             return UNCLASSIFIED_ERROR;
//         else
//         {
//             m_r = M / m + 1;
//             n_r = N / n + 1;
//             p_r = P / p + 1;
//             max_r = MAX(m_r, n_r);
//             max_r = MAX(max_r, p_r);
//             width = max_r / sub_domain + 1;
//         }
//         break;

//     case 2:
//         m = (int)(0.5 + (double)(((double)M) * ((double)size) / ((double)N)));
//         if (!m)
//             m = 1;
//         while (m > 0)
//         {
//             n = size / m;
//             if (m * n == size)
//                 break;
//             m--;
//         }
//         if (M > N && m < n)
//         {
//             int _m = m;
//             m = n;
//             n = _m;
//         }
//         if (m * n != size)
//             return UNCLASSIFIED_ERROR;
//         else
//         {
//             m_r = M / m + 1;
//             n_r = N / n + 1;
//             max_r = MAX(m_r, n_r);
//             width = max_r / sub_domain + 1;
//         }
//         break;

//     default:
//         return INVALID_INPUT;
//     }
//     *re_m = m;
//     *re_n = n;
//     *re_p = p;
//     *re_width = width;
//     return 0;
// }

int RT0_2d(double x, double y, int ind, double *phi_x, double *phi_y) {
  switch (ind) {
  case 0:
    *phi_x = (x + 1.) / 4.;
    *phi_y = 0.;
    break;
  case 1:
    *phi_x = 0.;
    *phi_y = (y + 1.) / 4.;
    break;
  case 2:
    *phi_x = (x - 1.) / 4.;
    *phi_y = 0.;
    break;
  case 3:
    *phi_x = 0.;
    *phi_y = (y - 1.) / 4.;
    break;
  default:
    return INVALID_INPUT;
  }
  return 0;
}

int RT0_3d(double x, double y, double z, int ind, double *phi_x, double *phi_y, double *phi_z) {
  switch (ind) {
  case 0:
    *phi_x = (x + 1.) / 8.;
    *phi_y = 0.;
    *phi_z = 0.;
    break;
  case 1:
    *phi_x = 0.;
    *phi_y = (y + 1.) / 8.;
    *phi_z = 0.;
    break;
  case 2:
    *phi_x = 0.;
    *phi_y = 0.;
    *phi_z = (z + 1.) / 8.;
    break;
  case 3:
    *phi_x = (x - 1.) / 8.;
    *phi_y = 0.;
    *phi_z = 0.;
    break;
  case 4:
    *phi_x = 0.;
    *phi_y = (y - 1.) / 8.;
    *phi_z = 0.;
    break;
  case 5:
    *phi_x = 0.;
    *phi_y = 0.;
    *phi_z = (z - 1.) / 8.;
    break;
  default:
    return INVALID_INPUT;
  }
  return 0;
}

int div_RT0_2d(double x, double y, int ind, double *div_phi) {
  switch (ind) {
  case 0:
  case 1:
  case 2:
  case 3:
    *div_phi = 0.25;
    break;
  default:
    return INVALID_INPUT;
  }
  return 0;
}

int div_RT1_2d(double x, double y, double z, int ind, double *div_phi) {
  switch (ind) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
    *div_phi = 0.125;
    break;
  default:
    return INVALID_INPUT;
  }
  return 0;
}

int Q0_2d(double x, double y, double *q0) {
  *q0 = 0.25;
  return 0;
}

int Q0_3d(double x, double y, double z, double *q0) {
  *q0 = 0.125;
  return 0;
}

/*
#define QUAD_ORDER 3
const double quad_cord[QUAD_ORDER] = {-0.77459667, 0.0, 0.77459667};
const double quad_wght[QUAD_ORDER] = {0.55555556, 0.88888888, 0.55555556};

int get_elem_mats(void)
{
    double x, y, z, phi_x_row, phi_y_row, phi_z_row, phi_x_col, phi_y_col, phi_z_col, q0, div_phi_row, div_phi_col;
    int ind_row, ind_col, i, j, k;

    for (ind_row = 0; ind_row < RT0_2D_NUM; ++ind_row)
    {
        for (ind_col = 0; ind_col < RT0_2D_NUM; ++ind_col)
        {
            for (i = 0; i < QUAD_ORDER; ++i)
            {
                x = quad_cord[i];
                for (j = 0; j < QUAD_ORDER; ++j)
                {
                    y = quad_cord[j];
                    RT0_2d(x, y, ind_row, &phi_x_row, &phi_y_row);
                    RT0_2d(x, y, ind_col, &phi_x_col, &phi_y_col);
                    div_RT0_2d(x, y, ind_row, &div_phi_row);
                    div_RT0_2d(x, y, ind_col, &div_phi_col);
                    elem_stiff_2d[ind_row][ind_col] += (phi_x_row * phi_x_col + phi_y_row * phi_y_col) * quad_wght[i] * quad_wght[j];
                    elem_divdiv_2d[ind_row][ind_col] += div_phi_row * div_phi_col * quad_wght[i] * quad_wght[j];
                }
            }
        }
        for (i = 0; i < QUAD_ORDER; ++i)
        {
            x = quad_cord[i];
            for (j = 0; j < QUAD_ORDER; ++j)
            {
                y = quad_cord[j];
                div_RT0_2d(x, y, ind_row, &div_phi_row);
                Q0_2d(x, y, &q0);
                elem_stiff_2d[ind_row][Q0_NUM + RT0_2D_NUM - 1] += div_phi_row * q0 * quad_wght[i] * quad_wght[j];
                elem_stiff_2d[Q0_NUM + RT0_2D_NUM - 1][ind_row] += q0 * div_phi_row * quad_wght[i] * quad_wght[j];
            }
        }
    }

    for (ind_row = 0; ind_row < RT0_3D_NUM; ++ind_row)
    {
        for (ind_col = 0; ind_col < RT0_3D_NUM; ++ind_col)
        {
            for (i = 0; i < QUAD_ORDER; ++i)
            {
                x = quad_cord[i];
                for (j = 0; j < QUAD_ORDER; ++j)
                {
                    y = quad_cord[j];
                    for (k = 0; k < QUAD_ORDER; ++k)
                    {
                        z = quad_cord[k];
                        RT0_3d(x, y, z, ind_row, &phi_x_row, &phi_y_row, &phi_z_row);
                        RT0_3d(x, y, z, ind_row, &phi_x_col, &phi_y_col, &phi_z_col);
                        div_RT0_3d(x, y, z, ind_row, &div_phi_row);
                        div_RT0_3d(x, y, z, ind_col, &div_phi_col);
                        elem_stiff_3d[ind_row][ind_col] += (phi_x_row * phi_x_col + phi_y_row * phi_y_col + phi_z_row * phi_z_col) * quad_wght[i] * quad_wght[j] * quad_wght[k];
                        elem_divdiv_3d[ind_row][ind_col] += div_phi_row * div_phi_col * quad_wght[i] * quad_wght[j] * quad_wght[k];
                    }
                }
            }
        }
        for (i = 0; i < QUAD_ORDER; ++i)
        {
            x = quad_cord[i];
            for (j = 0; j < QUAD_ORDER; ++j)
            {
                y = quad_cord[j];
                for (k = 0; k < QUAD_ORDER; ++k)
                {
                    z = quad_cord[k];
                    div_RT0_3d(x, y, z, ind_row, &div_phi_row);
                    Q0_3d(x, y, z, &q0);
                    elem_stiff_3d[ind_row][Q0_NUM + RT0_3D_NUM - 1] += div_phi_row * q0 * quad_wght[i] * quad_wght[j] * quad_wght[k];
                    elem_stiff_3d[Q0_NUM + RT0_3D_NUM - 1][ind_row] += q0 * div_phi_row * quad_wght[i] * quad_wght[j] * quad_wght[k];
                }
            }
        }
    }
}
*/

double u2d(double x, double y) {
  return cos(M_PI * x) * cos(M_PI * y);
}

double dx_u2d(double x, double y) {
  return -M_PI * sin(M_PI * x) * cos(M_PI * y);
}

double dy_u2d(double x, double y) {
  return -M_PI * cos(M_PI * x) * sin(M_PI * y);
}

double u3d(double x, double y, double z) {
  return cos(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
}

double dx_u3d(double x, double y, double z) {
  return -M_PI * sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
}

double dy_u3d(double x, double y, double z) {
  return -M_PI * cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
}

double dz_u3d(double x, double y, double z) {
  return -M_PI * cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
}

double kappa1(double x, double y) {
  return 1.0;
}

double source1(double x, double y) {
  return 2.0 * M_PI * M_PI * cos(M_PI * x) * cos(M_PI * y);
}

double kappa2(double x, double y, double z) {
  return 1.0;
}

double source2(double x, double y, double z) {
  return 3.0 * M_PI * M_PI * cos(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
}

double kappa3(double x, double y) {
  return 4.0 + sin(M_PI * x) + sin(2.0 * M_PI * y);
}

double source3(double x, double y)
// \[Pi]^2 Cos[\[Pi] x] (2 Cos[2 \[Pi] y] Sin[\[Pi] y]+Cos[\[Pi] y] (8+3 Sin[\[Pi] x]+2 Sin[2 \[Pi] y]))
{
  return M_PI * M_PI * cos(M_PI * x) * (2.0 * cos(2.0 * M_PI * y) * sin(M_PI * y) + cos(M_PI * y) * (8.0 + 3.0 * sin(M_PI * x) + 2.0 * sin(2 * M_PI * y)));
}

double kappa4(double x, double y, double z) {
  return 4.0 + sin(M_PI * x) + sin(2.0 * M_PI * y) + cos(0.5 * M_PI * z);
}

double source4(double x, double y, double z)
// 1/2 \[Pi]^2 Cos[\[Pi] x] (4 Cos[2 \[Pi] y] Cos[\[Pi] z] Sin[\[Pi] y]+Cos[\[Pi] y] (Cos[\[Pi] z] (24+6 Cos[(\[Pi] z)/2]+8 Sin[\[Pi] x]+6 Sin[2 \[Pi] y])-Sin[(\[Pi] z)/2] Sin[\[Pi] z])){
{
  return 0.5 * M_PI * M_PI * cos(M_PI * x) * (4.0 * cos(2.0 * M_PI * y) * cos(M_PI * z) * sin(M_PI * y) + cos(M_PI * y) * (cos(M_PI * z) * (24.0 + 6.0 * cos(0.5 * M_PI * z) + 8.0 * sin(M_PI * x) + 6.0 * sin(2.0 * M_PI * y)) - sin(0.5 * M_PI * z) * sin(M_PI * z)));
}

double source5(double x, double y) {
  if (0.0 <= x && x < 0.5 && 0.0 <= y && y < 0.5)
    return 1.0;
  if (0.5 <= x && x < 1.0 && 0.5 <= y && y < 1.0)
    return -1.0;
  if (0.0 <= x && x < 0.5 && 0.5 <= y && y < 1.0)
    return -0.5;
  if (0.5 <= x && x < 1.0 && 0.0 <= y && y < 0.5)
    return 0.5;
  return -1.0;
}

double source6(double x, double y, double z) {
  if (0.0 <= x && x < 0.5 && 0.0 <= y && y < 0.5)
    return 1.0;
  if (0.5 <= x && x < 1.0 && 0.5 <= y && y < 1.0)
    return -1.0;
  if (0.0 <= x && x < 0.5 && 0.5 <= y && y < 1.0)
    return -0.5;
  if (0.5 <= x && x < 1.0 && 0.0 <= y && y < 0.5)
    return 0.5;
  return -1.0;
}