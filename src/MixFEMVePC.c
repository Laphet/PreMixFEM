#include "MixFEMVePC.h"
#include "mpi.h"
#include "petscdm.h"
#include "petscdmda.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petsclog.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscpctypes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petsctime.h"
#include "petscvec.h"
#include <math.h>

#define NEG -1
#define MED 0
#define POS 1
#define _P_LEFT 0
#define _P_RIGHT 1
#define _P_DOWN 2
#define _P_UP 3
#define MAX_POS_2D 4
#define _P_BACK 4
#define _P_FRONT 5
#define MAX_POS_3D 6
#define SS_I 1
// Default snapshot space, boundary delta.
#define SS_II 2
// Simple snapshot space without solving snapshot problems.
#define NIL 1.0e-12

PetscErrorCode PC_init(PCCtx **init_ctx, ModelContext m_ctx, int M, int N, int P, int o_s, int s_d, int e_n, int i_l, PetscBool turn_off_A_0, PetscBool turn_off_A_i, PetscScalar robin_alpha) {
  PetscFunctionBeginUser;
  PCCtx *s_ctx;
  PetscCall(PetscNew(&s_ctx));
  *init_ctx = s_ctx;
  PetscScalar coarse_elem_p_num;
  PetscCheck(o_s >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid over-sampling layers=%d!\n", o_s);
  s_ctx->over_sampling = o_s;
  PetscCheck(s_d >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid subdomains=%d!\n", s_d);
  s_ctx->sub_domains = s_d;
  PetscCheck(e_n >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid eigenvector number=%d!\n", e_n);
  s_ctx->eigen_num = e_n;
  PetscCheck(i_l >= -1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid incomplete decomposition level=%d!\n", i_l);
  s_ctx->icc_level = i_l;
  PetscCheck(robin_alpha >= -NIL, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid Robin BC alpha=%.5f!\n", robin_alpha);

  s_ctx->A_0_off = turn_off_A_0;
  s_ctx->A_i_off = turn_off_A_i;
  s_ctx->robin_alpha = robin_alpha;
  s_ctx->m_ctx = m_ctx;
  s_ctx->M = M;
  s_ctx->N = N;
  s_ctx->H_x = m_ctx.L / (double)M;
  s_ctx->H_y = m_ctx.W / (double)N;
  switch (s_ctx->m_ctx.dim) {
  case 2:
    s_ctx->P = 0;
    s_ctx->H_z = 0.0;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, s_ctx->M, s_ctx->N, PETSC_DECIDE, PETSC_DECIDE, 1, s_ctx->over_sampling + 1, NULL, NULL, &(s_ctx->dm_os)));
    // Note the stencil width here is over_sampling + 1.
    break;
  case 3:
    s_ctx->P = P;
    s_ctx->H_z = m_ctx.H / (double)P;
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, s_ctx->M, s_ctx->N, s_ctx->P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, s_ctx->over_sampling + 1, NULL, NULL, NULL, &(s_ctx->dm_os)));
    break;
  default:
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid DIM=%d!\n", m_ctx.dim);
    break;
  }
  PetscCall(DMSetUp(s_ctx->dm_os));
  PetscCall(DMCreateGlobalVector(s_ctx->dm_os, &(s_ctx->v_kappa)));
  PetscInt m, n, p, *lx, *ly, *lz, i;
  PetscCall(DMDAGetInfo(s_ctx->dm_os, NULL, NULL, NULL, NULL, &m, &n, &p, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Processors in each direction: X=%d, Y=%d, Z=%d.\n", m, n, p));
  if ((m == 1 || n == 1 || p == 1) && s_ctx->sub_domains == 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The subdomain may not be appropriate (too long/wide/high), reset subdomain from %d to 2.\n", s_ctx->sub_domains));
    s_ctx->sub_domains = 2;
  }

  if (s_ctx->m_ctx.dim == 2)
    coarse_elem_p_num = s_ctx->sub_domains * s_ctx->sub_domains;
  if (s_ctx->m_ctx.dim == 3)
    coarse_elem_p_num = s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  s_ctx->coarse_elem_p_num = coarse_elem_p_num;

  PetscCall(PetscMalloc1(m, &lx));
  // Need to be freed.
  PetscCall(PetscMalloc1(n, &ly));
  // Need to be freed.
  for (i = 0; i < m; ++i)
    lx[i] = s_ctx->sub_domains;
  for (i = 0; i < n; ++i)
    ly[i] = s_ctx->sub_domains;
  switch (s_ctx->m_ctx.dim) {
  case 2:
    s_ctx->coarse_total_nx = s_ctx->sub_domains * m;
    s_ctx->coarse_total_ny = s_ctx->sub_domains * n;
    s_ctx->coarse_total_nz = 0;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, s_ctx->coarse_total_nx, s_ctx->coarse_total_ny, m, n, s_ctx->eigen_num, 1, lx, ly, &(s_ctx->dm_coarse_sp)));
    PetscCall(DMSetUp(s_ctx->dm_coarse_sp));
    break;
  case 3:
    PetscCall(PetscMalloc1(p, &lz));
    // Need to be freed.
    for (i = 0; i < p; ++i)
      lz[i] = s_ctx->sub_domains;
    s_ctx->coarse_total_nx = s_ctx->sub_domains * m;
    s_ctx->coarse_total_ny = s_ctx->sub_domains * n;
    s_ctx->coarse_total_nz = s_ctx->sub_domains * p;
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, s_ctx->coarse_total_nx, s_ctx->coarse_total_ny, s_ctx->coarse_total_nz, m, n, p, s_ctx->eigen_num, 1, lx, ly, lz, &(s_ctx->dm_coarse_sp)));
    PetscCall(DMSetUp(s_ctx->dm_coarse_sp));
    PetscCall(PetscFree(lz));
    break;
  default:
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid DIM=%d!\n", m_ctx.dim);
    break;
  }
  PetscCall(PetscFree(ly));
  PetscCall(PetscFree(lx));

  PetscCall(PetscLogStageRegister("su-0", &s_ctx->su0));
  PetscCall(PetscLogStageRegister("su-1", &s_ctx->su1));
  PetscCall(PetscLogStageRegister("su-2", &s_ctx->su2));
  PetscCall(PetscLogStageRegister("av-0", &s_ctx->av0));
  PetscCall(PetscLogStageRegister("av-1", &s_ctx->av1));
  PetscCall(PetscArrayzero(&s_ctx->su[0], 3));
  PetscCall(PetscArrayzero(&s_ctx->av[0], 2));

  //   PetscCall(PetscClassIdRegister("Set-up", &s_ctx->su));
  //   PetscCall(PetscLogEventRegister("su-0", s_ctx->su, &s_ctx->su0));
  //   PetscCall(PetscLogEventRegister("su-1", s_ctx->su, &s_ctx->su1));
  //   PetscCall(PetscLogEventRegister("su-2", s_ctx->su, &s_ctx->su2));
  //   PetscCall(PetscClassIdRegister("Apply-vec", &s_ctx->av));
  //   PetscCall(PetscLogEventRegister("av-0", s_ctx->av, &s_ctx->av0));
  //   PetscCall(PetscLogEventRegister("av-1", s_ctx->av, &s_ctx->av1));

  //   PetscCall(PetscMalloc1(coarse_elem_p_num, &s_ctx->A_i)); // Need to be freed.
  PetscCall(PetscMalloc1(s_ctx->eigen_num, &s_ctx->ms_bases));
  PetscCall(PetscMalloc1(s_ctx->coarse_elem_p_num, &s_ctx->ksp_i));
  PetscCall(PetscMalloc1(s_ctx->coarse_elem_p_num, &s_ctx->eigen_max));
  PetscCall(PetscMalloc1(s_ctx->coarse_elem_p_num, &s_ctx->eigen_min));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_startx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_starty));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_startz));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_lenx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_leny));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_lenz));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_p_startx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_p_starty));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_p_startz));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_p_lenx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_p_leny));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_p_lenz));
  //   PetscCall(DMCreateMatrix(s_ctx->dm_coarse_sp, &(s_ctx->A_0)));
  PetscFunctionReturn(0);
}

PetscErrorCode PC_final_default(PCCtx **s_ctx_) {
  PetscFunctionBeginUser;
  PCCtx *s_ctx = *s_ctx_;
  PetscCall(PetscFree(s_ctx->eigen_max));
  PetscCall(PetscFree(s_ctx->eigen_min));
  PetscCall(VecDestroy(&(s_ctx->v_kappa)));
  PetscCall(KSPDestroy(&(s_ctx->ksp_0)));
  //   PetscCall(PetscFree(s_ctx->A_i));
  PetscCall(PetscFree(s_ctx->ms_bases));
  PetscCall(PetscFree(s_ctx->ksp_i));
  PetscCall(PetscFree(s_ctx->coarse_startx));
  PetscCall(PetscFree(s_ctx->coarse_starty));
  PetscCall(PetscFree(s_ctx->coarse_startz));
  PetscCall(PetscFree(s_ctx->coarse_lenx));
  PetscCall(PetscFree(s_ctx->coarse_leny));
  PetscCall(PetscFree(s_ctx->coarse_lenz));
  PetscCall(PetscFree(s_ctx->coarse_p_startx));
  PetscCall(PetscFree(s_ctx->coarse_p_starty));
  PetscCall(PetscFree(s_ctx->coarse_p_startz));
  PetscCall(PetscFree(s_ctx->coarse_p_lenx));
  PetscCall(PetscFree(s_ctx->coarse_p_leny));
  PetscCall(PetscFree(s_ctx->coarse_p_lenz));
  PetscCall(PetscFree(s_ctx));
  s_ctx = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PC_final(PCCtx **s_ctx_) {
  PetscFunctionBeginUser;
  PCCtx *s_ctx = *s_ctx_;
  PetscCall(DMDestroy(&(s_ctx->dm_os)));
  PetscCall(DMDestroy(&(s_ctx->dm_coarse_sp)));
  PetscInt i, dof;
  for (i = 0; i < s_ctx->coarse_elem_p_num; ++i) {
    PetscCall(KSPDestroy(&(s_ctx->ksp_i[i])));
  }
  for (dof = 0; dof < s_ctx->eigen_num; ++dof) {
    PetscCall(VecDestroy(&(s_ctx->ms_bases[dof])));
  }
  PetscCall(PC_final_default(s_ctx_));
  PetscFunctionReturn(0);
}

PetscErrorCode PC_setup(PC pc) {
  PetscFunctionBeginUser;
  PCCtx *s_ctx;
  PetscLogDouble time_tmp;
  PetscCall(PCShellGetContext(pc, &s_ctx));
  /*********************************************************************
    Get the boundaries of each coarse subdomain.
  *********************************************************************/
  PetscCall(PetscLogStagePush(s_ctx->su0));
  PetscCall(PetscTime(&time_tmp));
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, i, j;
  PetscInt startx_, starty_, startz_, nx_, ny_, nz_;
  PetscScalar meas_elem, meas_face_yz, meas_face_zx, meas_face_xy;
  PetscCall(DMDAGetCorners(s_ctx->dm_os, &startx, &starty, &startz, &nx, &ny, &nz));
  s_ctx->coarse_startx[0] = startx;
  s_ctx->coarse_starty[0] = starty;
  s_ctx->coarse_lenx[0] = (nx / s_ctx->sub_domains) + (nx % s_ctx->sub_domains > 0);
  s_ctx->coarse_leny[0] = (ny / s_ctx->sub_domains) + (ny % s_ctx->sub_domains > 0);
  for (i = 1; i < s_ctx->sub_domains; ++i) {
    s_ctx->coarse_startx[i] = s_ctx->coarse_startx[i - 1] + s_ctx->coarse_lenx[i - 1];
    s_ctx->coarse_lenx[i] = (nx / s_ctx->sub_domains) + (nx % s_ctx->sub_domains > i);
    s_ctx->coarse_starty[i] = s_ctx->coarse_starty[i - 1] + s_ctx->coarse_leny[i - 1];
    s_ctx->coarse_leny[i] = (ny / s_ctx->sub_domains) + (ny % s_ctx->sub_domains > i);
  }
  for (i = 0; i < s_ctx->sub_domains; ++i) {
    s_ctx->coarse_p_startx[i] = s_ctx->coarse_startx[i] - s_ctx->over_sampling >= 0 ? s_ctx->coarse_startx[i] - s_ctx->over_sampling : 0;
    s_ctx->coarse_p_lenx[i] = s_ctx->coarse_startx[i] + s_ctx->coarse_lenx[i] + s_ctx->over_sampling <= s_ctx->M ? s_ctx->coarse_startx[i] + s_ctx->coarse_lenx[i] + s_ctx->over_sampling - s_ctx->coarse_p_startx[i] : s_ctx->M - s_ctx->coarse_p_startx[i];
    s_ctx->coarse_p_starty[i] = s_ctx->coarse_starty[i] - s_ctx->over_sampling >= 0 ? s_ctx->coarse_starty[i] - s_ctx->over_sampling : 0;
    s_ctx->coarse_p_leny[i] = s_ctx->coarse_starty[i] + s_ctx->coarse_leny[i] + s_ctx->over_sampling <= s_ctx->N ? s_ctx->coarse_starty[i] + s_ctx->coarse_leny[i] + s_ctx->over_sampling - s_ctx->coarse_p_starty[i] : s_ctx->N - s_ctx->coarse_p_starty[i];
  }
  if (s_ctx->m_ctx.dim == 3) {
    s_ctx->coarse_startz[0] = startz;
    s_ctx->coarse_lenz[0] = (nz / s_ctx->sub_domains) + (nz % s_ctx->sub_domains > 0);
    for (i = 1; i < s_ctx->sub_domains; ++i) {
      s_ctx->coarse_startz[i] = s_ctx->coarse_startz[i - 1] + s_ctx->coarse_lenz[i - 1];
      s_ctx->coarse_lenz[i] = (nz / s_ctx->sub_domains) + (nz % s_ctx->sub_domains > i);
    }
    for (i = 0; i < s_ctx->sub_domains; ++i) {
      s_ctx->coarse_p_startz[i] = s_ctx->coarse_startz[i] - s_ctx->over_sampling >= 0 ? s_ctx->coarse_startz[i] - s_ctx->over_sampling : 0;
      s_ctx->coarse_p_lenz[i] = s_ctx->coarse_startz[i] + s_ctx->coarse_lenz[i] + s_ctx->over_sampling <= s_ctx->P ? s_ctx->coarse_startz[i] + s_ctx->coarse_lenz[i] + s_ctx->over_sampling - s_ctx->coarse_p_startz[i] : s_ctx->P - s_ctx->coarse_p_startz[i];
    }
  }
  if (s_ctx->m_ctx.dim == 2) {
    meas_elem = s_ctx->H_x * s_ctx->H_y;
  }
  if (s_ctx->m_ctx.dim == 3) {
    meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
    meas_face_yz = s_ctx->H_y * s_ctx->H_z;
    meas_face_zx = s_ctx->H_z * s_ctx->H_x;
    meas_face_xy = s_ctx->H_x * s_ctx->H_y;
  }

  /*********************************************************************
    Get A_i.
  *********************************************************************/
  Mat A_i;
  Vec v_kappa_loc;
  PetscScalar **arr_kappa_2d, ***arr_kappa_3d, avg_kappa_e;
  PetscInt coarse_elem_p_x, coarse_elem_p_y, coarse_elem_p_z, coarse_elem_p;
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &v_kappa_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm_os, s_ctx->v_kappa, INSERT_VALUES, v_kappa_loc));
  if (s_ctx->m_ctx.dim == 2) {
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_2d));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = coarse_elem_p / s_ctx->sub_domains;
      startx = s_ctx->coarse_p_startx[coarse_elem_p_x];
      nx = s_ctx->coarse_p_lenx[coarse_elem_p_x];
      starty = s_ctx->coarse_p_starty[coarse_elem_p_y];
      ny = s_ctx->coarse_p_leny[coarse_elem_p_y];
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny, nx * ny, 5, NULL, &A_i));
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex) {
          // We first handle homogeneous Neumann BCs.
          if (ex >= startx + 1)
          // Inner x-direction edges.
          {
            PetscInt row[2], col[2];
            row[0] = (ey - starty) * nx + ex - startx - 1;
            row[1] = (ey - starty) * nx + ex - startx;
            col[0] = (ey - starty) * nx + ex - startx - 1;
            col[1] = (ey - starty) * nx + ex - startx;
            PetscScalar val_A[2][2];
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex - 1] + 1.0 / arr_kappa_2d[ey][ex]);
            val_A[0][0] = s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            val_A[0][1] = -s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            val_A[1][0] = -s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            val_A[1][1] = s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          } else if (ex >= 1)
          // Left boundary on the coarse element.
          {
            PetscInt row[1], col[1];
            PetscScalar val_A[1][1];
            row[0] = (ey - starty) * nx + ex - startx;
            // avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex - 1] + 1.0 / arr_kappa_2d[ey][ex]);
            avg_kappa_e = 2.0 * arr_kappa_2d[ey][ex];
            col[0] = (ey - starty) * nx + ex - startx;
            val_A[0][0] = s_ctx->H_y * s_ctx->H_y / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * s_ctx->H_y) * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ex + 1 == startx + nx && ex + 1 != s_ctx->M) // Right boundary on the coarse element.
          {
            PetscInt row[1], col[1];
            PetscScalar val_A[1][1];
            row[0] = (ey - starty) * nx + ex - startx;
            // avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex] + 1.0 / arr_kappa_2d[ey][ex + 1]); // Need to be careful.
            avg_kappa_e = 2.0 * arr_kappa_2d[ey][ex];
            col[0] = (ey - starty) * nx + ex - startx;
            val_A[0][0] = s_ctx->H_y * s_ctx->H_y / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * s_ctx->H_y) * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ey >= starty + 1)
          // Inner y-direction edges.
          {
            PetscInt row[2], col[2];
            row[0] = (ey - starty - 1) * nx + ex - startx;
            row[1] = (ey - starty) * nx + ex - startx;
            col[0] = (ey - starty - 1) * nx + ex - startx;
            col[1] = (ey - starty) * nx + ex - startx;
            PetscScalar val_A[2][2];
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey - 1][ex] + 1.0 / arr_kappa_2d[ey][ex]);
            val_A[0][0] = s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            val_A[0][1] = -s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            val_A[1][0] = -s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            val_A[1][1] = s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          } else if (ey >= 1)
          // Down boundary on the coarse element
          {
            PetscInt row[1], col[1];
            PetscScalar val_A[1][1];
            row[0] = (ey - starty) * nx + ex - startx;
            // avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey - 1][ex] + 1.0 / arr_kappa_2d[ey][ex]);
            avg_kappa_e = 2.0 * arr_kappa_2d[ey][ex];
            col[0] = (ey - starty) * nx + ex - startx;
            val_A[0][0] = s_ctx->H_x * s_ctx->H_x / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * s_ctx->H_x) * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ey + 1 == starty + ny && ey + 1 != s_ctx->N)
          // Up boundary on the coarse element.
          {
            PetscInt row[1], col[1];
            PetscScalar val_A[1][1];
            row[0] = (ey - starty) * nx + ex - startx;
            // avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex] + 1.0 / arr_kappa_2d[ey + 1][ex]);
            avg_kappa_e = 2.0 * arr_kappa_2d[ey][ex];
            col[0] = (ey - starty) * nx + ex - startx;
            val_A[0][0] = s_ctx->H_x * s_ctx->H_x / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * s_ctx->H_x) * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
        }
      PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
      PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
      PetscCall(KSPCreate(PETSC_COMM_SELF, &(s_ctx->ksp_i[coarse_elem_p])));
      PetscCall(KSPSetOperators(s_ctx->ksp_i[coarse_elem_p], A_i, A_i));
      {
        PC pc_;
        PetscCall(KSPSetType(s_ctx->ksp_i[coarse_elem_p], KSPPREONLY));
        PetscCall(KSPGetPC(s_ctx->ksp_i[coarse_elem_p], &pc_));
        if (s_ctx->icc_level == -1) {
          PetscCall(PCSetType(pc_, PCCHOLESKY));
          PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERCHOLMOD));
          //   PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERPETSC));
          PetscCall(PCSetOptionsPrefix(pc_, "Ai_"));
          PetscCall(PCSetFromOptions(pc_));
        } else {
          PetscCall(PCSetType(pc_, PCICC));
          PetscCall(PCFactorSetLevels(pc_, s_ctx->icc_level));
        }
      }
      PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_i[coarse_elem_p], PETSC_TRUE));
      PetscCall(KSPSetUp(s_ctx->ksp_i[coarse_elem_p]));
      PetscCall(MatDestroy(&A_i));
    }
  }
  if (s_ctx->m_ctx.dim == 3) {
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_3d));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
      coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
      startx = s_ctx->coarse_p_startx[coarse_elem_p_x];
      nx = s_ctx->coarse_p_lenx[coarse_elem_p_x];
      starty = s_ctx->coarse_p_starty[coarse_elem_p_y];
      ny = s_ctx->coarse_p_leny[coarse_elem_p_y];
      startz = s_ctx->coarse_p_startz[coarse_elem_p_z];
      nz = s_ctx->coarse_p_lenz[coarse_elem_p_z];
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx * ny * nz, 7, NULL, &A_i));

      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          for (ex = startx; ex < startx + nx; ++ex) {
            // We first handle homogeneous Neumann BCs.
            if (ex >= startx + 1)
            // Inner x-direction edges.
            {
              PetscInt row[2], col[2];
              PetscScalar val_A[2][2];
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1;
              row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1;
              col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex - 1] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            } else if (ex >= 1)
            // Left boundary on the coarse element.
            {
              PetscInt row[1], col[1];
              PetscScalar val_A[1][1];
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              //   avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex - 1] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              avg_kappa_e = 2.0 * arr_kappa_3d[ez][ey][ex];
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = meas_face_yz * meas_face_yz / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * meas_face_yz) * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
            }
            if (ex + 1 == startx + nx && ex + 1 != s_ctx->M)
            // Right boundary on the coarse element.
            {
              PetscInt row[1], col[1];
              PetscScalar val_A[1][1];
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              //   avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex + 1]); // Need to be careful.
              avg_kappa_e = 2.0 * arr_kappa_3d[ez][ey][ex];
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = meas_face_yz * meas_face_yz / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * meas_face_yz) * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
            }

            if (ey >= starty + 1)
            // Inner y-direction edges.
            {
              PetscInt row[2], col[2];
              PetscScalar val_A[2][2];
              row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex - startx;
              row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex - startx;
              col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey - 1][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            } else if (ey >= 1)
            // Down boundary on the coarse element
            {
              PetscInt row[1], col[1];
              PetscScalar val_A[1][1];
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              //   avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey - 1][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              avg_kappa_e = 2.0 * arr_kappa_3d[ez][ey][ex];
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = meas_face_zx * meas_face_zx / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * meas_face_zx) * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
            }
            if (ey + 1 == starty + ny && ey + 1 != s_ctx->N)
            // Up boundary on the coarse element.
            {
              PetscInt row[1], col[1];
              PetscScalar val_A[1][1];
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              //   avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex] + 1.0 / arr_kappa_3d[ez][ey + 1][ex]);
              avg_kappa_e = 2.0 * arr_kappa_3d[ez][ey][ex];
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = meas_face_zx * meas_face_zx / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * meas_face_zx) * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
            }

            if (ez >= startz + 1)
            // Inner z-direction edges.
            {
              PetscInt row[2], col[2];
              PetscScalar val_A[2][2];
              row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex - startx;
              row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex - startx;
              col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez - 1][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            } else if (ez >= 1)
            // Back boundary on the coarse element
            {
              PetscInt row[1], col[1];
              PetscScalar val_A[1][1];
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              //   avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez - 1][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              avg_kappa_e = 2.0 * arr_kappa_3d[ez][ey][ex];
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = meas_face_xy * meas_face_xy / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * meas_face_xy) * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
            }
            if (ez + 1 == startz + nz && ez + 1 != s_ctx->P)
            // Front boundary on the coarse element.
            {
              PetscInt row[1], col[1];
              PetscScalar val_A[1][1];
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              //   avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex] + 1.0 / arr_kappa_3d[ez + 1][ey][ex]);
              avg_kappa_e = 2.0 * arr_kappa_3d[ez][ey][ex];
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = meas_face_xy * meas_face_xy / (meas_elem + avg_kappa_e * s_ctx->robin_alpha * meas_face_xy) * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
            }
          }

      PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
      PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
      PetscCall(KSPCreate(PETSC_COMM_SELF, &(s_ctx->ksp_i[coarse_elem_p])));
      PetscCall(KSPSetOperators(s_ctx->ksp_i[coarse_elem_p], A_i, A_i));
      {
        PC pc_;
        PetscCall(KSPSetType(s_ctx->ksp_i[coarse_elem_p], KSPPREONLY));
        PetscCall(KSPGetPC(s_ctx->ksp_i[coarse_elem_p], &pc_));
        if (s_ctx->icc_level == -1) {
          PetscCall(PCSetType(pc_, PCCHOLESKY));
          PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERCHOLMOD));
          PetscCall(PCSetOptionsPrefix(pc_, "Ai_"));
          PetscCall(PCSetFromOptions(pc_));
        } else {
          PetscCall(PCSetType(pc_, PCICC));
          PetscCall(PCFactorSetLevels(pc_, s_ctx->icc_level));
        }
      }
      PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_i[coarse_elem_p], PETSC_TRUE));
      PetscCall(KSPSetUp(s_ctx->ksp_i[coarse_elem_p]));
      PetscCall(MatDestroy(&A_i));
    }
  }
  PetscCall(PetscLogStagePop());
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->su[0] -= time_tmp;

  /*********************************************************************
    Get coarse info.
  *********************************************************************/
  PetscCall(PetscLogStagePush(s_ctx->su1));
  PetscCall(PetscTime(&time_tmp));
  Mat A_i_inner, M_i;
  Vec diag_M_i;
  PetscInt dof;
  for (dof = 0; dof < s_ctx->eigen_num; ++dof) {
    PetscCall(DMCreateLocalVector(s_ctx->dm_os, &(s_ctx->ms_bases[dof])));
  }
  if (s_ctx->m_ctx.dim == 2) {
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      PetscScalar **arr_M_i_2d;
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = coarse_elem_p / s_ctx->sub_domains;
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx_ * ny_, nx_ * ny_, 5, NULL, &A_i_inner));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx_ * ny_, nx_ * ny_, 1, NULL, &M_i));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nx_ * ny_, &diag_M_i));
      PetscCall(VecGetArray2d(diag_M_i, ny_, nx_, 0, 0, &arr_M_i_2d));
      for (ey = starty_; ey < starty_ + ny_; ++ey) {
        for (ex = startx_; ex < startx_ + nx_; ++ex) {
          if (ex >= startx_ + 1) {
            PetscInt row_inner[2], col_inner[2];
            row_inner[0] = (ey - starty_) * nx_ + ex - startx_ - 1;
            row_inner[1] = (ey - starty_) * nx_ + ex - startx_;
            col_inner[0] = (ey - starty_) * nx_ + ex - startx_ - 1;
            col_inner[1] = (ey - starty_) * nx_ + ex - startx_;
            PetscScalar val_A[2][2];
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex - 1] + 1.0 / arr_kappa_2d[ey][ex]);
            val_A[0][0] = s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            val_A[0][1] = -s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            val_A[1][0] = -s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            val_A[1][1] = s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row_inner[0], 2, &col_inner[0], &val_A[0][0], ADD_VALUES));
          }
          if (ey >= starty_ + 1) {
            PetscInt row_inner[2], col_inner[2];
            row_inner[0] = (ey - starty_ - 1) * nx_ + ex - startx_;
            row_inner[1] = (ey - starty_) * nx_ + ex - startx_;
            col_inner[0] = (ey - starty_ - 1) * nx_ + ex - startx_;
            col_inner[1] = (ey - starty_) * nx_ + ex - startx_;
            PetscScalar val_A[2][2];
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey - 1][ex] + 1.0 / arr_kappa_2d[ey][ex]);
            val_A[0][0] = s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            val_A[0][1] = -s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            val_A[1][0] = -s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            val_A[1][1] = s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row_inner[0], 2, &col_inner[0], &val_A[0][0], ADD_VALUES));
          }
        }
        PetscCall(PetscArraycpy(&arr_M_i_2d[ey - starty_][0], &arr_kappa_2d[ey][startx_], nx_));
        // Memory copy kappa to M_i, for efficiency.
      }
      PetscCall(VecRestoreArray2d(diag_M_i, ny_, nx_, 0, 0, &arr_M_i_2d));
      PetscCall(VecScale(diag_M_i, meas_elem * 2.0 * M_PI * M_PI / (nx_ * nx_ * s_ctx->H_x * s_ctx->H_x + ny_ * ny_ * s_ctx->H_y * s_ctx->H_y)));
      // Scaling to real mass matrix.
      PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
      PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(MatDiagonalSet(M_i, diag_M_i, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(M_i, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(M_i, MAT_FINAL_ASSEMBLY));
      PetscCall(VecDestroy(&diag_M_i));
      // Destory diag_M_i here.

      EPS eps;
      PetscInt nconv;
      PetscScalar eig_val;
      Vec eig_vec;

      PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
      PetscCall(EPSSetOperators(eps, A_i_inner, M_i));
      PetscCall(EPSSetProblemType(eps, EPS_GHEP));
      PetscCall(EPSSetDimensions(eps, s_ctx->eigen_num, PETSC_DEFAULT, PETSC_DEFAULT));
      PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));

      ST st;
      PetscCall(EPSGetST(eps, &st));
      PetscCall(STSetType(st, STSINVERT));
      PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
      PetscCall(EPSSetTarget(eps, -1.0));
      PetscCall(EPSSetFromOptions(eps));

      PetscCall(EPSSolve(eps));
      PetscCall(EPSGetConverged(eps, &nconv));
      PetscCheck(nconv >= s_ctx->eigen_num, PETSC_COMM_WORLD, PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors! (nconv=%d, eigen_num=%d)\n", nconv, s_ctx->eigen_num);
      for (j = 0; j < s_ctx->eigen_num; ++j) {
        PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
        if (j == 0)
          s_ctx->eigen_min[coarse_elem_p] = eig_val;
        if (j == s_ctx->eigen_num - 1)
          s_ctx->eigen_max[coarse_elem_p] = eig_val;
        PetscScalar **arr_ms_bases_loc, **arr_eig_vec;
        PetscCall(DMDAVecGetArray(s_ctx->dm_os, s_ctx->ms_bases[j], &arr_ms_bases_loc));
        PetscCall(VecGetArray2d(eig_vec, ny_, nx_, 0, 0, &arr_eig_vec));
        for (ey = starty_; ey < starty_ + ny_; ++ey)
          PetscCall(PetscArraycpy(&arr_ms_bases_loc[ey][startx_], &arr_eig_vec[ey - starty_][0], nx_));
        PetscCall(VecRestoreArray2d(eig_vec, ny_, nx_, 0, 0, &arr_eig_vec));
        PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, s_ctx->ms_bases[j], &arr_ms_bases_loc));
      }
      // Do some clean.
      PetscCall(VecDestroy(&eig_vec));
      PetscCall(EPSDestroy(&eps));
      PetscCall(MatDestroy(&A_i_inner));
      PetscCall(MatDestroy(&M_i));
    }
  }

  if (s_ctx->m_ctx.dim == 3) {
    PetscScalar ***arr_M_i_3d;
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
      coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      startz_ = s_ctx->coarse_startz[coarse_elem_p_z];
      nz_ = s_ctx->coarse_lenz[coarse_elem_p_z];

      // 7 point stencil.
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nx_ * ny_ * nz_, &diag_M_i));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx_ * ny_ * nz_, nx_ * ny_ * nz_, 7, NULL, &A_i_inner));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx_ * ny_ * nz_, nx_ * ny_ * nz_, 1, NULL, &M_i));
      PetscCall(VecGetArray3d(diag_M_i, nz_, ny_, nx_, 0, 0, 0, &arr_M_i_3d));
      for (ez = startz_; ez < startz_ + nz_; ++ez)
        for (ey = starty_; ey < starty_ + ny_; ++ey) {
          for (ex = startx_; ex < startx_ + nx_; ++ex) {
            if (ex >= startx_ + 1) {
              PetscInt row[2], col[2];
              PetscScalar val_A[2][2];
              row[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_ - 1;
              row[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              col[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_ - 1;
              col[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex - 1] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            }
            if (ey >= starty_ + 1) {
              PetscInt row[2], col[2];
              PetscScalar val_A[2][2];
              row[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_ - 1) * nx_ + ex - startx_;
              row[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              col[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_ - 1) * nx_ + ex - startx_;
              col[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey - 1][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            }
            if (ez >= startz_ + 1) {
              PetscInt row[2], col[2];
              PetscScalar val_A[2][2];
              row[0] = (ez - startz_ - 1) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              row[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              col[0] = (ez - startz_ - 1) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              col[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez - 1][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
              val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            }
          }
          PetscCall(PetscArraycpy(&arr_M_i_3d[ez - startz_][ey - starty_][0], &arr_kappa_3d[ez][ey][startx_], nx_));
          // Memory copy kappa to M_i, for efficiency.
        }
      PetscCall(VecRestoreArray3d(diag_M_i, nz_, ny_, nx_, 0, 0, 0, &arr_M_i_3d));
      PetscCall(VecScale(diag_M_i, meas_elem * 3.0 * M_PI * M_PI / (nx_ * nx_ * s_ctx->H_x * s_ctx->H_x + ny_ * ny_ * s_ctx->H_y * s_ctx->H_y + nz_ * nz_ * s_ctx->H_z * s_ctx->H_z)));
      PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
      PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(MatDiagonalSet(M_i, diag_M_i, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(M_i, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(M_i, MAT_FINAL_ASSEMBLY));
      PetscCall(VecDestroy(&diag_M_i));

      EPS eps;
      PetscInt nconv;
      PetscScalar eig_val;
      Vec eig_vec;

      PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
      PetscCall(EPSSetOperators(eps, A_i_inner, M_i));
      PetscCall(EPSSetProblemType(eps, EPS_GHEP));
      PetscCall(EPSSetDimensions(eps, s_ctx->eigen_num, PETSC_DEFAULT, PETSC_DEFAULT));
      PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));

      //   PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE));
      ST st;
      PetscCall(EPSGetST(eps, &st));
      PetscCall(STSetType(st, STSINVERT));
      //   PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
      PetscCall(EPSSetTarget(eps, -NIL));
      PetscCall(EPSSetFromOptions(eps));

      //   Vec is[4];
      //   PetscScalar ***arr_is[4], x, y, z;
      //   for (i = 0; i < 4; ++i)
      //     PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz_ * ny_ * nz_, &is[i]));
      //   PetscCall(VecSet(is[0], 1.0));
      //   PetscCall(VecGetArray3d(is[1], nz_, ny_, nx_, 0, 0, 0, &arr_is[1]));
      //   PetscCall(VecGetArray3d(is[2], nz_, ny_, nx_, 0, 0, 0, &arr_is[2]));
      //   PetscCall(VecGetArray3d(is[3], nz_, ny_, nx_, 0, 0, 0, &arr_is[3]));
      //   for (ez = 0; ez < nz_; ++ez)
      //     for (ey = 0; ey < ny_; ++ey)
      //       for (ex = 0; ex < nx_; ++ex) {
      //         x = 0.5 * (double)(2 * ex + 1) / (double)nx_ * M_PI;
      //         y = 0.5 * (double)(2 * ey + 1) / (double)ny_ * M_PI;
      //         z = 0.5 * (double)(2 * ez + 1) / (double)nz_ * M_PI;
      //         arr_is[1][ez][ey][ex] = cos(x);
      //         arr_is[2][ez][ey][ex] = cos(y);
      //         arr_is[3][ez][ey][ex] = cos(z);
      //       }
      //   PetscCall(VecRestoreArray3d(is[1], nz_, ny_, nx_, 0, 0, 0, &arr_is[1]));
      //   PetscCall(VecRestoreArray3d(is[2], nz_, ny_, nx_, 0, 0, 0, &arr_is[2]));
      //   PetscCall(VecRestoreArray3d(is[3], nz_, ny_, nx_, 0, 0, 0, &arr_is[3]));
      //   PetscCall(EPSSetInitialSpace(eps, 4, &is[0]));

      PetscCall(EPSSolve(eps));
      PetscCall(EPSGetConverged(eps, &nconv));
      PetscCheck(nconv >= s_ctx->eigen_num, PETSC_COMM_WORLD, PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors! (nconv=%d, eigen_num=%d)\n", nconv, s_ctx->eigen_num);
      for (j = 0; j < s_ctx->eigen_num; ++j) {
        PetscScalar ***arr_ms_bases_loc, ***arr_eig_vec;
        PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
        if (j == 0)
          s_ctx->eigen_min[coarse_elem_p] = eig_val;
        if (j == s_ctx->eigen_num - 1)
          s_ctx->eigen_max[coarse_elem_p] = eig_val;
        // PetscPrintf(PETSC_COMM_WORLD, "j=%d, eig_val=%.5f.\n", j, eig_val);
        PetscCall(DMDAVecGetArray(s_ctx->dm_os, s_ctx->ms_bases[j], &arr_ms_bases_loc));
        PetscCall(VecGetArray3d(eig_vec, nz_, ny_, nx_, 0, 0, 0, &arr_eig_vec));
        for (ez = startz_; ez < startz_ + nz_; ++ez)
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            PetscCall(PetscArraycpy(&arr_ms_bases_loc[ez][ey][startx_], &arr_eig_vec[ez - startz_][ey - starty_][0], nx_));
        PetscCall(VecRestoreArray3d(eig_vec, nz_, ny_, nx_, 0, 0, 0, &arr_eig_vec));
        PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, s_ctx->ms_bases[j], &arr_ms_bases_loc));
      }
      // Do some clean.
      PetscCall(VecDestroy(&eig_vec));
      PetscCall(EPSDestroy(&eps));
      //   for (i = 0; i < 4; ++i)
      //     PetscCall(VecDestroy(&is[i]));
      PetscCall(MatDestroy(&A_i_inner));
      PetscCall(MatDestroy(&M_i));
    }
  }
  Vec dummy_ms_bases_glo;
  PetscCall(DMCreateGlobalVector(s_ctx->dm_os, &dummy_ms_bases_glo));
  for (dof = 0; dof < s_ctx->eigen_num; ++dof) {
    PetscCall(DMLocalToGlobal(s_ctx->dm_os, s_ctx->ms_bases[dof], INSERT_VALUES, dummy_ms_bases_glo));
    PetscCall(DMGlobalToLocal(s_ctx->dm_os, dummy_ms_bases_glo, INSERT_VALUES, s_ctx->ms_bases[dof]));
  }
  PetscCall(VecDestroy(&dummy_ms_bases_glo));

  PetscCall(PetscLogStagePop());
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->su[1] -= time_tmp;

  /*********************************************************************
  // Construct A_0, note here we need to write communications by ourself.
  *********************************************************************/
  PetscCall(PetscLogStagePush(s_ctx->su2));
  PetscCall(PetscTime(&time_tmp));
  Mat A_0;
  PetscCall(DMCreateMatrix(s_ctx->dm_coarse_sp, &A_0));
  PetscInt coarse_elem_startx, coarse_elem_starty, coarse_elem_startz, coarse_elem_nx, coarse_elem_ny, coarse_elem_nz, dof_row, dof_col;
  PetscScalar A0_val;
  PetscCall(DMDAGetCorners(s_ctx->dm_coarse_sp, &coarse_elem_startx, &coarse_elem_starty, &coarse_elem_startz, &coarse_elem_nx, &coarse_elem_ny, &coarse_elem_nz));
  if (s_ctx->m_ctx.dim == 2) {
    PetscCheck(coarse_elem_nx == s_ctx->sub_domains && coarse_elem_ny == s_ctx->sub_domains, PETSC_COMM_WORLD, PETSC_ERR_USER, "Something wrong here, sub_domains=%d, nx=%d, ny=%d.\n", s_ctx->sub_domains, coarse_elem_nx, coarse_elem_ny);
  } else {
    PetscCheck(coarse_elem_nx == s_ctx->sub_domains && coarse_elem_ny == s_ctx->sub_domains && coarse_elem_nz == s_ctx->sub_domains, PETSC_COMM_WORLD, PETSC_ERR_USER, "Something wrong here, sub_domains=%d, nx=%d, ny=%d, nz=%d.\n", s_ctx->sub_domains, coarse_elem_nx, coarse_elem_ny, coarse_elem_nz);
  }

  if (s_ctx->m_ctx.dim == 2) {
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = coarse_elem_p / s_ctx->sub_domains;
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      for (dof_row = 0; dof_row < s_ctx->eigen_num; ++dof_row) {
        PetscScalar **arr_ms_bases_row;
        PetscCall(DMDAVecGetArray(s_ctx->dm_os, s_ctx->ms_bases[dof_row], &arr_ms_bases_row));
        MatStencil stencil_row = {.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y, .c = dof_row};
        for (dof_col = 0; dof_col < s_ctx->eigen_num; ++dof_col) {
          PetscScalar **arr_ms_bases_col;
          PetscCall(DMDAVecGetArray(s_ctx->dm_os, s_ctx->ms_bases[dof_col], &arr_ms_bases_col));
          MatStencil stencil_col = {.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y, .c = dof_col};
          A0_val = 0.0;
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              if (ex >= startx_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex - 1] + 1.0 / arr_kappa_2d[ey][ex]);
                A0_val += s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e * (arr_ms_bases_row[ey][ex - 1] - arr_ms_bases_row[ey][ex]) * (arr_ms_bases_col[ey][ex - 1] - arr_ms_bases_col[ey][ex]);
              } else if (ex >= 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex - 1] + 1.0 / arr_kappa_2d[ey][ex]);
                A0_val += s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e * arr_ms_bases_row[ey][ex] * arr_ms_bases_col[ey][ex];
              }
              if (ex + 1 == startx_ + nx_ && ex + 1 != s_ctx->M) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex] + 1.0 / arr_kappa_2d[ey][ex + 1]);
                A0_val += s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e * arr_ms_bases_row[ey][ex] * arr_ms_bases_col[ey][ex];
              }

              if (ey >= starty_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey - 1][ex] + 1.0 / arr_kappa_2d[ey][ex]);
                A0_val += s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e * (arr_ms_bases_row[ey - 1][ex] - arr_ms_bases_row[ey][ex]) * (arr_ms_bases_col[ey - 1][ex] - arr_ms_bases_col[ey][ex]);
              } else if (ey >= 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey - 1][ex] + 1.0 / arr_kappa_2d[ey][ex]);
                A0_val += s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e * arr_ms_bases_row[ey][ex] * arr_ms_bases_col[ey][ex];
              }
              if (ey + 1 == starty_ + ny_ && ey + 1 != s_ctx->N) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex] + 1.0 / arr_kappa_2d[ey + 1][ex]);
                A0_val += s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e * arr_ms_bases_row[ey][ex] * arr_ms_bases_col[ey][ex];
              }
            }
          PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));

          if (stencil_row.i != 0) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x - 1, .j = coarse_elem_starty + coarse_elem_p_y, .c = dof_col};
            A0_val = 0.0;
            for (ey = starty_; ey < starty_ + ny_; ++ey) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][startx_ - 1] + 1.0 / arr_kappa_2d[ey][startx_]);
              A0_val -= s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e * arr_ms_bases_col[ey][startx_] * arr_ms_bases_col[ey][startx_ - 1];
            }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.i != s_ctx->coarse_total_nx - 1) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x + 1, .j = coarse_elem_starty + coarse_elem_p_y, .c = dof_col};
            A0_val = 0.0;
            for (ey = starty_; ey < starty_ + ny_; ++ey) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][startx_ + nx_ - 1] + 1.0 / arr_kappa_2d[ey][startx_ + nx_]);
              A0_val -= s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e * arr_ms_bases_row[ey][startx_ + nx_ - 1] * arr_ms_bases_col[ey][startx_ + nx_];
            }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.j != 0) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y - 1, .c = dof_col};
            A0_val = 0.0;
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[starty_ - 1][ex] + 1.0 / arr_kappa_2d[starty_][ex]);
              A0_val -= s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e * arr_ms_bases_row[starty_][ex] * arr_ms_bases_col[starty_ - 1][ex];
            }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.j != s_ctx->coarse_total_ny - 1) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y + 1, .c = dof_col};
            A0_val = 0.0;
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[starty_ + ny_ - 1][ex] + 1.0 / arr_kappa_2d[starty_ + ny_][ex]);
              A0_val -= s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e * arr_ms_bases_row[starty_ + ny_ - 1][ex] * arr_ms_bases_col[starty_ + ny_][ex];
            }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }
          PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, s_ctx->ms_bases[dof_col], &arr_ms_bases_col));
        }
        PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, s_ctx->ms_bases[dof_row], &arr_ms_bases_row));
      }
    }
  }

  if (s_ctx->m_ctx.dim == 3) {
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
      coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      startz_ = s_ctx->coarse_startz[coarse_elem_p_z];
      nz_ = s_ctx->coarse_lenz[coarse_elem_p_z];
      for (dof_row = 0; dof_row < s_ctx->eigen_num; ++dof_row) {
        PetscScalar ***arr_ms_bases_row;
        PetscCall(DMDAVecGetArray(s_ctx->dm_os, s_ctx->ms_bases[dof_row], &arr_ms_bases_row));
        MatStencil stencil_row = {.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y, .k = coarse_elem_startz + coarse_elem_p_z, .c = dof_row};
        for (dof_col = 0; dof_col < s_ctx->eigen_num; ++dof_col) {
          PetscScalar ***arr_ms_bases_col;
          PetscCall(DMDAVecGetArray(s_ctx->dm_os, s_ctx->ms_bases[dof_col], &arr_ms_bases_col));
          MatStencil stencil_col = {.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y, .k = coarse_elem_startz + coarse_elem_p_z, .c = dof_col};
          A0_val = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ey = starty_; ey < starty_ + ny_; ++ey)
              for (ex = startx_; ex < startx_ + nx_; ++ex) {
                if (ex >= startx_ + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex - 1] + 1.0 / arr_kappa_3d[ez][ey][ex]);
                  A0_val += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * (arr_ms_bases_row[ez][ey][ex - 1] - arr_ms_bases_row[ez][ey][ex]) * (arr_ms_bases_col[ez][ey][ex - 1] - arr_ms_bases_col[ez][ey][ex]);
                } else if (ex >= 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex - 1] + 1.0 / arr_kappa_3d[ez][ey][ex]);
                  A0_val += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][ex] * arr_ms_bases_col[ez][ey][ex];
                }
                if (ex + 1 == startx_ + nx_ && ex + 1 != s_ctx->M) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex + 1]);
                  A0_val += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][ex] * arr_ms_bases_col[ez][ey][ex];
                }

                if (ey >= starty_ + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey - 1][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
                  A0_val += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * (arr_ms_bases_row[ez][ey - 1][ex] - arr_ms_bases_row[ez][ey][ex]) * (arr_ms_bases_col[ez][ey - 1][ex] - arr_ms_bases_col[ez][ey][ex]);
                } else if (ey >= 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey - 1][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
                  A0_val += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][ex] * arr_ms_bases_col[ez][ey][ex];
                }
                if (ey + 1 == starty_ + ny_ && ey + 1 != s_ctx->N) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex] + 1.0 / arr_kappa_3d[ez][ey + 1][ex]);
                  A0_val += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][ex] * arr_ms_bases_col[ez][ey][ex];
                }

                if (ez >= startz_ + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez - 1][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
                  A0_val += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * (arr_ms_bases_row[ez - 1][ey][ex] - arr_ms_bases_row[ez][ey][ex]) * (arr_ms_bases_col[ez - 1][ey][ex] - arr_ms_bases_col[ez][ey][ex]);
                } else if (ez >= 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez - 1][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);
                  A0_val += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][ex] * arr_ms_bases_col[ez][ey][ex];
                }
                if (ez + 1 == startz_ + nz_ && ez + 1 != s_ctx->P) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex] + 1.0 / arr_kappa_3d[ez + 1][ey][ex]);
                  A0_val += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][ex] * arr_ms_bases_col[ez][ey][ex];
                }
              }
          PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));

          if (stencil_row.i != 0) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x - 1, .j = coarse_elem_starty + coarse_elem_p_y, .k = coarse_elem_startz + coarse_elem_p_z, .c = dof_col};
            A0_val = 0.0;
            for (ez = startz_; ez < startz_ + nz_; ++ez)
              for (ey = starty_; ey < starty_ + ny_; ++ey) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][startx_ - 1] + 1.0 / arr_kappa_3d[ez][ey][startx_]);
                A0_val -= meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][startx_] * arr_ms_bases_col[ez][ey][startx_ - 1];
              }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.i != s_ctx->coarse_total_nx - 1) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x + 1, .j = coarse_elem_starty + coarse_elem_p_y, .k = coarse_elem_startz + coarse_elem_p_z, .c = dof_col};
            A0_val = 0.0;
            for (ez = startz_; ez < startz_ + nz_; ++ez)
              for (ey = starty_; ey < starty_ + ny_; ++ey) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][startx_ + nx_ - 1] + 1.0 / arr_kappa_3d[ez][ey][startx_ + nx_]);
                A0_val -= meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][ey][startx_ + nx_ - 1] * arr_ms_bases_col[ez][ey][startx_ + nx_];
              }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.j != 0) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y - 1, .k = coarse_elem_startz + coarse_elem_p_z, .c = dof_col};
            A0_val = 0.0;
            for (ez = startz_; ez < startz_ + nz_; ++ez)
              for (ex = startx_; ex < startx_ + nx_; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][starty_ - 1][ex] + 1.0 / arr_kappa_3d[ez][starty_][ex]);
                A0_val -= meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][starty_][ex] * arr_ms_bases_col[ez][starty_ - 1][ex];
              }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.j != s_ctx->coarse_total_ny - 1) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y + 1, .k = coarse_elem_startz + coarse_elem_p_z, .c = dof_col};
            A0_val = 0.0;
            for (ez = startz_; ez < startz_ + nz_; ++ez)
              for (ex = startx_; ex < startx_ + nx_; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][starty_ + ny_ - 1][ex] + 1.0 / arr_kappa_3d[ez][starty_ + ny_][ex]);
                A0_val -= meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_row[ez][starty_ + ny_ - 1][ex] * arr_ms_bases_col[ez][starty_ + ny_][ex];
              }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.k != 0) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y, .k = coarse_elem_startz + coarse_elem_p_z - 1, .c = dof_col};
            A0_val = 0.0;
            for (ey = starty_; ey < starty_ + ny_; ++ey)
              for (ex = startx_; ex < startx_ + nx_; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[startz_ - 1][ey][ex] + 1.0 / arr_kappa_3d[startz_][ey][ex]);
                A0_val -= meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_row[startz_][ey][ex] * arr_ms_bases_col[startz_ - 1][ey][ex];
              }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }

          if (stencil_row.k != s_ctx->coarse_total_nz - 1) {
            stencil_col = (MatStencil){.i = coarse_elem_startx + coarse_elem_p_x, .j = coarse_elem_starty + coarse_elem_p_y, .k = coarse_elem_startz + coarse_elem_p_z + 1, .c = dof_col};
            A0_val = 0.0;
            for (ey = starty_; ey < starty_ + ny_; ++ey)
              for (ex = startx_; ex < startx_ + nx_; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[startz_ + nz_ - 1][ey][ex] + 1.0 / arr_kappa_3d[startz_ + nz_][ey][ex]);
                A0_val -= meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_row[startz_ + nz_ - 1][ey][ex] * arr_ms_bases_col[startz_ + nz_][ey][ex];
              }
            PetscCall(MatSetValuesStencil(A_0, 1, &stencil_row, 1, &stencil_col, &A0_val, INSERT_VALUES));
          }
          PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, s_ctx->ms_bases[dof_col], &arr_ms_bases_col));
        }
        PetscCall(DMDAVecGetArray(s_ctx->dm_os, s_ctx->ms_bases[dof_row], &arr_ms_bases_row));
      }
    }
  }

  PetscCall(MatAssemblyBegin(A_0, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_0, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(A_0, NIL));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &s_ctx->ksp_0));
  PetscCall(KSPSetOperators(s_ctx->ksp_0, A_0, A_0));
  {
    PC pc_;
    PetscCall(KSPSetType(s_ctx->ksp_0, KSPPREONLY));
    PetscCall(KSPGetPC(s_ctx->ksp_0, &pc_));
    PetscCall(PCSetType(pc_, PCLU));
    PetscCall(PCFactorSetShiftType(pc_, MAT_SHIFT_NONZERO));
    PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERMUMPS));
    PetscCall(PCSetOptionsPrefix(pc_, "A0_"));
    PetscCall(PCSetFromOptions(pc_));
  }
  PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_0, PETSC_TRUE));
  PetscCall(KSPSetUp(s_ctx->ksp_0));
  PetscCall(MatDestroy(&A_0));

  /*********************************************************************
    Clean.
  *********************************************************************/
  if (s_ctx->m_ctx.dim == 2)
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_2d));
  else if (s_ctx->m_ctx.dim == 3)
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_3d));
  PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &v_kappa_loc));

  PetscCall(PetscLogStagePop());
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->su[2] -= time_tmp;
  PetscFunctionReturn(0);
}

PetscErrorCode PC_apply_vec_i(PC pc, Vec x, Vec y) {
  // Input x, Return y.
  // We assume y is proper, and the result would be added to y.
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(y));
  PetscLogDouble time_tmp;
  PCCtx *s_ctx;
  PetscCall(PCShellGetContext(pc, &s_ctx));
  PetscCall(PetscLogStagePush(s_ctx->av0));
  PetscCall(PetscTime(&time_tmp));

  PetscInt ey, ez, startx, starty, startz, nx, ny, nz;
  PetscInt coarse_elem_p_x, coarse_elem_p_y, coarse_elem_p_z, coarse_elem_p;
  Vec x_loc, y_loc, temp_loc;
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &x_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm_os, x, INSERT_VALUES, x_loc));
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &y_loc));
  PetscCall(VecZeroEntries(y_loc));
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &temp_loc));

  if (s_ctx->m_ctx.dim == 2) {
    PetscScalar **arr_x, **arr_temp;
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, x_loc, &arr_x));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      Vec rhs, sol;
      PetscScalar **arr_rhs, **arr_sol;
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = coarse_elem_p / s_ctx->sub_domains;
      startx = s_ctx->coarse_p_startx[coarse_elem_p_x];
      nx = s_ctx->coarse_p_lenx[coarse_elem_p_x];
      starty = s_ctx->coarse_p_starty[coarse_elem_p_y];
      ny = s_ctx->coarse_p_leny[coarse_elem_p_y];
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, ny * nx, &rhs));
      PetscCall(VecDuplicate(rhs, &sol));
      PetscCall(VecGetArray2d(rhs, ny, nx, 0, 0, &arr_rhs));
      for (ey = starty; ey < starty + ny; ++ey)
        PetscCall(PetscArraycpy(&arr_rhs[ey - starty][0], &arr_x[ey][startx], nx));
      PetscCall(VecRestoreArray2d(rhs, ny, nx, 0, 0, &arr_rhs));
      PetscCall(KSPSolve(s_ctx->ksp_i[coarse_elem_p], rhs, sol));

      VecDestroy(&rhs);

      PetscCall(VecGetArray2d(sol, ny, nx, 0, 0, &arr_sol));
      PetscCall(VecZeroEntries(temp_loc));
      PetscCall(DMDAVecGetArray(s_ctx->dm_os, temp_loc, &arr_temp));
      for (ey = starty; ey < starty + ny; ++ey)
        PetscCall(PetscArraycpy(&arr_temp[ey][startx], &arr_sol[ey - starty][0], nx));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, temp_loc, &arr_temp));

      VecDestroy(&sol);

      PetscCall(VecAXPY(y_loc, 1.0, temp_loc));
    }
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm_os, x_loc, &arr_x));
  }
  if (s_ctx->m_ctx.dim == 3) {
    PetscScalar ***arr_x, ***arr_temp;
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, x_loc, &arr_x));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      Vec rhs, sol;
      PetscScalar ***arr_rhs, ***arr_sol;
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
      coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
      startx = s_ctx->coarse_p_startx[coarse_elem_p_x];
      nx = s_ctx->coarse_p_lenx[coarse_elem_p_x];
      starty = s_ctx->coarse_p_starty[coarse_elem_p_y];
      ny = s_ctx->coarse_p_leny[coarse_elem_p_y];
      startz = s_ctx->coarse_p_startz[coarse_elem_p_z];
      nz = s_ctx->coarse_p_lenz[coarse_elem_p_z];
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &rhs));
      PetscCall(VecDuplicate(rhs, &sol));
      PetscCall(VecGetArray3d(rhs, nz, ny, nx, 0, 0, 0, &arr_rhs));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          PetscCall(PetscArraycpy(&arr_rhs[ez - startz][ey - starty][0], &arr_x[ez][ey][startx], nx));
      PetscCall(VecRestoreArray3d(rhs, nz, ny, nx, 0, 0, 0, &arr_rhs));
      PetscCall(KSPSolve(s_ctx->ksp_i[coarse_elem_p], rhs, sol));

      VecDestroy(&rhs);

      PetscCall(VecGetArray3d(sol, nz, ny, nx, 0, 0, 0, &arr_sol));
      PetscCall(VecZeroEntries(temp_loc));
      PetscCall(DMDAVecGetArray(s_ctx->dm_os, temp_loc, &arr_temp));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          PetscCall(PetscArraycpy(&arr_temp[ez][ey][startx], &arr_sol[ez - startz][ey - starty][0], nx));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, temp_loc, &arr_temp));

      VecDestroy(&sol);
      PetscCall(VecAXPY(y_loc, 1.0, temp_loc));
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(s_ctx->dm_os, x_loc, &arr_x));
  }

  PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &x_loc));
  PetscCall(DMLocalToGlobal(s_ctx->dm_os, y_loc, ADD_VALUES, y));
  PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &y_loc));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->av[0] -= time_tmp;
  PetscFunctionReturn(0);
}

PetscErrorCode PC_apply_vec_0(PC pc, Vec x, Vec y) {
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(y));
  PetscLogDouble time_tmp;
  PCCtx *s_ctx;
  PetscCall(PCShellGetContext(pc, &s_ctx));
  PetscCall(PetscLogStagePush(s_ctx->av1));
  PetscCall(PetscTime(&time_tmp));

  Vec x_loc, rhs0, sol0, temp_os_loc;
  PetscInt startx_, starty_, startz_, nx_, ny_, nz_, ey, ez;
  PetscInt coarse_elem_startx, coarse_elem_starty, coarse_elem_startz, coarse_elem_p_x, coarse_elem_p_y, coarse_elem_p_z, coarse_elem_p;
  PetscCall(DMDAGetCorners(s_ctx->dm_coarse_sp, &coarse_elem_startx, &coarse_elem_starty, &coarse_elem_startz, NULL, NULL, NULL));

  PetscCall(DMGetLocalVector(s_ctx->dm_os, &x_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm_os, x, INSERT_VALUES, x_loc));
  PetscCall(DMCreateGlobalVector(s_ctx->dm_coarse_sp, &rhs0));
  PetscCall(DMCreateGlobalVector(s_ctx->dm_coarse_sp, &sol0));
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &temp_os_loc));

  if (s_ctx->m_ctx.dim == 2) {
    PetscScalar ***arr_rhs0, ***arr_sol0, **arr_temp_os, **arr_x, **arr_y;
    PetscCall(DMDAVecGetArrayDOF(s_ctx->dm_coarse_sp, rhs0, &arr_rhs0));
    PetscCall(DMDAVecGetArray(s_ctx->dm_os, x_loc, &arr_x));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = coarse_elem_p / s_ctx->sub_domains;
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      PetscCall(VecZeroEntries(temp_os_loc));
      PetscCall(DMDAVecGetArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
      for (ey = starty_; ey < starty_ + ny_; ++ey)
        PetscCall(PetscArraycpy(&arr_temp_os[ey][startx_], &arr_x[ey][startx_], nx_));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
      PetscCall(VecMDot(temp_os_loc, s_ctx->eigen_num, &s_ctx->ms_bases[0], &arr_rhs0[coarse_elem_starty + coarse_elem_p_y][coarse_elem_startx + coarse_elem_p_x][0]));
    }
    PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, x_loc, &arr_x));
    PetscCall(DMDAVecRestoreArrayDOF(s_ctx->dm_coarse_sp, rhs0, &arr_rhs0));
    PetscCall(KSPSolve(s_ctx->ksp_0, rhs0, sol0));

    PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &x_loc));
    PetscCall(VecDestroy(&rhs0));

    PetscCall(DMDAVecGetArrayDOF(s_ctx->dm_coarse_sp, sol0, &arr_sol0));
    PetscCall(DMDAVecGetArray(s_ctx->dm_os, y, &arr_y));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = coarse_elem_p / s_ctx->sub_domains;
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      PetscCall(VecZeroEntries(temp_os_loc));
      PetscCall(VecMAXPY(temp_os_loc, s_ctx->eigen_num, &arr_sol0[coarse_elem_starty + coarse_elem_p_y][coarse_elem_startx + coarse_elem_p_x][0], &s_ctx->ms_bases[0]));
      PetscCall(DMDAVecGetArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
      for (ey = starty_; ey < starty_ + ny_; ++ey)
        PetscCall(PetscArraycpy(&arr_y[ey][startx_], &arr_temp_os[ey][startx_], nx_));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
    }
    PetscCall(DMDAVecRestoreArrayDOF(s_ctx->dm_coarse_sp, sol0, &arr_sol0));
    PetscCall(VecDestroy(&sol0));
    PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &temp_os_loc));

    PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, y, &arr_y));
  }

  if (s_ctx->m_ctx.dim == 3) {
    PetscScalar ****arr_rhs0, ****arr_sol0, ***arr_temp_os, ***arr_x, ***arr_y;
    PetscCall(DMDAVecGetArrayDOF(s_ctx->dm_coarse_sp, rhs0, &arr_rhs0));
    PetscCall(DMDAVecGetArray(s_ctx->dm_os, x_loc, &arr_x));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
      coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      startz_ = s_ctx->coarse_startz[coarse_elem_p_z];
      nz_ = s_ctx->coarse_lenz[coarse_elem_p_z];
      PetscCall(VecZeroEntries(temp_os_loc));
      PetscCall(DMDAVecGetArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
      for (ez = startz_; ez < startz_ + nz_; ++ez)
        for (ey = starty_; ey < starty_ + ny_; ++ey)
          PetscCall(PetscArraycpy(&arr_temp_os[ez][ey][startx_], &arr_x[ez][ey][startx_], nx_));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
      PetscCall(VecMDot(temp_os_loc, s_ctx->eigen_num, &s_ctx->ms_bases[0], &arr_rhs0[coarse_elem_startz + coarse_elem_p_z][coarse_elem_starty + coarse_elem_p_y][coarse_elem_startx + coarse_elem_p_x][0]));
    }
    PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, x_loc, &arr_x));
    PetscCall(DMDAVecRestoreArrayDOF(s_ctx->dm_coarse_sp, rhs0, &arr_rhs0));
    PetscCall(KSPSolve(s_ctx->ksp_0, rhs0, sol0));

    PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &x_loc));
    PetscCall(VecDestroy(&rhs0));

    PetscCall(DMDAVecGetArrayDOF(s_ctx->dm_coarse_sp, sol0, &arr_sol0));
    PetscCall(DMDAVecGetArray(s_ctx->dm_os, y, &arr_y));
    for (coarse_elem_p = 0; coarse_elem_p < s_ctx->coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
      coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      startz_ = s_ctx->coarse_startz[coarse_elem_p_z];
      nz_ = s_ctx->coarse_lenz[coarse_elem_p_z];
      PetscCall(VecZeroEntries(temp_os_loc));
      PetscCall(VecMAXPY(temp_os_loc, s_ctx->eigen_num, &arr_sol0[coarse_elem_startz + coarse_elem_p_z][coarse_elem_starty + coarse_elem_p_y][coarse_elem_startx + coarse_elem_p_x][0], &s_ctx->ms_bases[0]));
      PetscCall(DMDAVecGetArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
      for (ez = startz_; ez < startz_ + nz_; ++ez)
        for (ey = starty_; ey < starty_ + ny_; ++ey)
          PetscCall(PetscArraycpy(&arr_y[ez][ey][startx_], &arr_temp_os[ez][ey][startx_], nx_));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, temp_os_loc, &arr_temp_os));
    }
    PetscCall(DMDAVecRestoreArrayDOF(s_ctx->dm_coarse_sp, sol0, &arr_sol0));
    PetscCall(VecDestroy(&sol0));
    PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &temp_os_loc));

    PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, y, &arr_y));
  }

  PetscCall(PetscLogStagePop());
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->av[1] -= time_tmp;
  PetscFunctionReturn(0);
}

PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y) {
  // input x, return y.
  PetscFunctionBeginUser;

  PCCtx *s_ctx;
  PetscCall(PCShellGetContext(pc, &s_ctx));
  if (!s_ctx->A_i_off && !s_ctx->A_0_off) {
    Vec temp;
    PetscCall(VecDuplicate(y, &temp));
    PetscCall(PC_apply_vec_i(pc, x, y));
    PetscCall(PC_apply_vec_0(pc, x, temp));
    PetscCall(VecAXPY(y, 1.0, temp));
    PetscCall(VecDestroy(&temp));
  }

  if (!s_ctx->A_i_off && s_ctx->A_0_off)
    PC_apply_vec_i(pc, x, y);
  if (s_ctx->A_i_off && !s_ctx->A_0_off)
    PC_apply_vec_0(pc, x, y);

  PetscFunctionReturn(0);
}

PetscErrorCode PC_apply_vec_hybrid(PC pc, Vec x, Vec y) {
  PetscFunctionBeginUser;
  PCCtx *s_ctx;
  PetscCall(PCShellGetContext(pc, &s_ctx));
  Mat A;
  PetscCall(PCGetOperators(pc, &A, NULL));
  Vec temp_1, temp_2, temp_3;
  PetscCall(VecDuplicate(y, &temp_1));
  PetscCall(VecDuplicate(y, &temp_2));
  PetscCall(VecDuplicate(y, &temp_3));
  PetscCall(PC_apply_vec_0(pc, x, y));
  PetscCall(MatMult(A, y, temp_1));
  PetscCall(VecAYPX(temp_1, -1.0, x));
  PetscCall(PC_apply_vec_i(pc, temp_1, temp_2));
  PetscCall(MatMult(A, temp_2, temp_1));
  PetscCall(PC_apply_vec_0(pc, temp_1, temp_3));
  PetscCall(VecDestroy(&temp_1));
  PetscCall(VecAYPX(temp_3, -1.0, temp_2));
  PetscCall(VecDestroy(&temp_2));
  PetscCall(VecAYPX(y, 1.0, temp_3));
  PetscCall(VecDestroy(&temp_3));
  PetscFunctionReturn(0);
}

PetscErrorCode PC_create_system(PCCtx *s_ctx, Vec v_source, Mat *A, Vec *rhs) {
  PetscFunctionBeginUser;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz;
  PetscScalar val_rhs, avg_kappa_e, meas_elem, meas_face_xy, meas_face_yz, meas_face_zx;
  Vec v_kappa_loc, v_source_loc, rhs_loc;

  PetscCall(DMCreateMatrix(s_ctx->dm_os, A));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(DMCreateGlobalVector(s_ctx->dm_os, rhs));
  PetscCall(DMDAGetCorners(s_ctx->dm_os, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &v_kappa_loc));
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &v_source_loc));
  PetscCall(DMGetLocalVector(s_ctx->dm_os, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm_os, s_ctx->v_kappa, INSERT_VALUES, v_kappa_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm_os, v_source, INSERT_VALUES, v_source_loc));
  PetscScalar **arr_kappa_2d, **arr_source_2d, **arr_rhs_2d, ***arr_kappa_3d, ***arr_source_3d, ***arr_rhs_3d;
  switch (s_ctx->m_ctx.dim) {
  case 2:
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_2d));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, v_source_loc, &arr_source_2d));
    PetscCall(DMDAVecGetArray(s_ctx->dm_os, rhs_loc, &arr_rhs_2d));
    meas_elem = s_ctx->H_x * s_ctx->H_y;
    for (ey = starty; ey < starty + ny; ++ey)
      for (ex = startx; ex < startx + nx; ++ex) {
        // We first handle homogeneous Neumann BCs.
        if (ex >= 1) {
          MatStencil row[2], col[2];
          PetscScalar val_A[2][2];
          row[0] = (MatStencil){.i = ex - 1, .j = ey};
          row[1] = (MatStencil){.i = ex, .j = ey};
          col[0] = (MatStencil){.i = ex - 1, .j = ey};
          col[1] = (MatStencil){.i = ex, .j = ey};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey][ex - 1] + 1.0 / arr_kappa_2d[ey][ex]);
          val_A[0][0] = s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
          val_A[0][1] = -s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
          val_A[1][0] = -s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
          val_A[1][1] = s_ctx->H_y * s_ctx->H_y / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
        }
        if (ey >= 1) {
          MatStencil row[2], col[2];
          PetscScalar val_A[2][2];
          row[0] = (MatStencil){.i = ex, .j = ey - 1};
          row[1] = (MatStencil){.i = ex, .j = ey};
          col[0] = (MatStencil){.i = ex, .j = ey - 1};
          col[1] = (MatStencil){.i = ex, .j = ey};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_2d[ey - 1][ex] + 1.0 / arr_kappa_2d[ey][ex]);
          val_A[0][0] = s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
          val_A[0][1] = -s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
          val_A[1][0] = -s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
          val_A[1][1] = s_ctx->H_x * s_ctx->H_x / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
        }
        {
          val_rhs = arr_source_2d[ey][ex] * meas_elem;
          arr_rhs_2d[ey][ex] += val_rhs;
        }
      }
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_2d));
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm_os, v_source_loc, &arr_source_2d));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, rhs_loc, &arr_rhs_2d));
    break;

  case 3:
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_3d));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm_os, v_source_loc, &arr_source_3d));
    PetscCall(DMDAVecGetArray(s_ctx->dm_os, rhs_loc, &arr_rhs_3d));
    meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
    meas_face_yz = s_ctx->H_y * s_ctx->H_z;
    meas_face_zx = s_ctx->H_z * s_ctx->H_x;
    meas_face_xy = s_ctx->H_x * s_ctx->H_y;
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex) {
          // We first handle homogeneous Neumann BCs.
          if (ex >= 1) {
            MatStencil row[2], col[2];
            PetscScalar val_A[2][2];
            row[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
            row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
            col[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
            col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey][ex - 1] + 1.0 / arr_kappa_3d[ez][ey][ex]);

            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ey >= 1) {
            MatStencil row[2], col[2];
            PetscScalar val_A[2][2];
            row[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
            row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
            col[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
            col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez][ey - 1][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);

            val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ez >= 1) {
            MatStencil row[2], col[2];
            PetscScalar val_A[2][2];
            row[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
            row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
            col[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
            col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[ez - 1][ey][ex] + 1.0 / arr_kappa_3d[ez][ey][ex]);

            val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          }
          {
            val_rhs = arr_source_3d[ez][ey][ex] * meas_elem;
            arr_rhs_3d[ez][ey][ex] += val_rhs;
          }
        }
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm_os, v_kappa_loc, &arr_kappa_3d));
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm_os, v_source_loc, &arr_source_3d));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm_os, rhs_loc, &arr_rhs_3d));
    break;
  default:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Invalid input!\n"));
    break;
  }
  PetscCall(DMLocalToGlobal(s_ctx->dm_os, rhs_loc, ADD_VALUES, *rhs));
  PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &v_kappa_loc));
  PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &v_source_loc));
  PetscCall(DMRestoreLocalVector(s_ctx->dm_os, &rhs_loc));
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
