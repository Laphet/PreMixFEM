#include "PreMixFEM_3D.h"
#include "petscdm.h"
#include "petscdmda.h"
#include "petscdmdatypes.h"
#include "petscerror.h"
#include "petsclog.h"
#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <stdlib.h>

PetscErrorCode PC_init(PCCtx **init_ctx, PetscScalar *dom, PetscInt *mesh, PetscScalar *fl_args, PetscInt *int_args, PetscBool *b_args) {

  PetscFunctionBeginUser;
  PetscCheck((dom[0] > 0.0 && dom[1] > 0.0 && dom[2] > 0.0), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Errors in dom=[%.5f, %.5f, %.5f].\n", dom[0], dom[1], dom[2]);
  PetscCheck((mesh[0] > 0 && mesh[1] > 0 && mesh[2] > 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Errors in mesh=[%d, %d, %d].\n", mesh[0], mesh[1], mesh[2]);
  PetscCheck(int_args[0] >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Error in oversampling_layers=%d.\n", int_args[0]);
  PetscCheck(int_args[1] >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Error in subdomains=%d.\n", int_args[1]);
  PetscCheck(int_args[2] >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Error in eigenvectors=%d for the level1 problem.\n", int_args[2]);
  PetscCheck(int_args[3] >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Error in eigenvectors=%d for the level2 problem.\n", int_args[3]);
  if (fl_args[0] < 0.0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set a negative eigenvalue threshold (%.5f) for the level1 problem, will use all eigenvectors (by replacing the bound with 1.0e+12).\n", fl_args[0]));
    fl_args[0] = 1.0 / NIL;
  }
  if (fl_args[1] < 0.0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set a negative eigenvalue threshold (%.5f) for the level2 problem, will use all eigenvectors (by replacing the bound with 1.0e+12).\n", fl_args[1]));
    fl_args[1] = 1.0 / NIL;
  }

  PCCtx *s_ctx = *init_ctx;
  s_ctx->L = dom[0];
  s_ctx->W = dom[1];
  s_ctx->H = dom[2];
  s_ctx->M = mesh[0];
  s_ctx->N = mesh[1];
  s_ctx->P = mesh[2];

  s_ctx->eigen_bd_lv1 = fl_args[0];
  s_ctx->eigen_bd_lv2 = fl_args[1];
  s_ctx->over_sampling = int_args[0];
  s_ctx->sub_domains = int_args[1];
  s_ctx->max_eigen_num_lv1 = int_args[2];
  s_ctx->max_eigen_num_lv2 = int_args[3];

  s_ctx->H_x = dom[0] / (double)mesh[0];
  s_ctx->H_y = dom[1] / (double)mesh[1];
  s_ctx->H_z = dom[2] / (double)mesh[2];

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, s_ctx->M, s_ctx->N, s_ctx->P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, s_ctx->over_sampling, NULL, NULL, NULL, &(s_ctx->dm)));
  // If oversampling=1, DMDA has a ghost point width=1 now, and this will change the construction of A_i in level-1.
  PetscCall(DMSetUp(s_ctx->dm));

  PetscInt m, n, p, i, coarse_elem_p_num;
  for (i = 0; i < DIM; ++i)
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &(s_ctx->kappa[i])));
  PetscCall(DMDAGetInfo(s_ctx->dm, NULL, NULL, NULL, NULL, &m, &n, &p, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Processors in each direction: X=%d, Y=%d, Z=%d.\n", m, n, p));
  if ((m == 1 || n == 1 || p == 1) && s_ctx->sub_domains == 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The subdomain may not be proper (too long/wide/high), reset subdomain from %d to 2.\n", s_ctx->sub_domains));
    s_ctx->sub_domains = 2;
  }
  coarse_elem_p_num = s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;

  PetscCall(PetscMalloc1(coarse_elem_p_num, &s_ctx->eigen_num_lv1));
  PetscCall(PetscMalloc1(coarse_elem_p_num, &s_ctx->eigen_max_lv1));
  PetscCall(PetscMalloc1(coarse_elem_p_num, &s_ctx->eigen_min_lv1));

  s_ctx->ms_bases = (Vec *)malloc(coarse_elem_p_num * sizeof(Vec));
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

  PetscFunctionReturn(0);
}

PetscErrorCode PC_setup(PC pc) {
  PetscFunctionBeginUser;
  PCCtx *s_ctx;
  PetscCall(PCShellGetContext(pc, &s_ctx));

  /*********************************************************************
    Get the boundaries of each coarse subdomain.
  *********************************************************************/
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, i, j;
  PetscInt startx_, starty_, startz_, nx_, ny_, nz_;
  PetscScalar meas_elem, meas_face_yz, meas_face_zx, meas_face_xy;
  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz, &proc_nx, &proc_ny, &proc_nz));
  s_ctx->coarse_startx[0] = proc_startx;
  s_ctx->coarse_starty[0] = proc_starty;
  s_ctx->coarse_startz[0] = proc_startz;
  s_ctx->coarse_lenx[0] = (proc_nx / s_ctx->sub_domains) + (proc_nx % s_ctx->sub_domains > 0);
  s_ctx->coarse_leny[0] = (proc_ny / s_ctx->sub_domains) + (proc_ny % s_ctx->sub_domains > 0);
  s_ctx->coarse_lenz[0] = (proc_nz / s_ctx->sub_domains) + (proc_nz % s_ctx->sub_domains > 0);
  for (i = 1; i < s_ctx->sub_domains; ++i) {
    s_ctx->coarse_startx[i] = s_ctx->coarse_startx[i - 1] + s_ctx->coarse_lenx[i - 1];
    s_ctx->coarse_lenx[i] = (proc_nx / s_ctx->sub_domains) + (proc_nx % s_ctx->sub_domains > i);
    s_ctx->coarse_starty[i] = s_ctx->coarse_starty[i - 1] + s_ctx->coarse_leny[i - 1];
    s_ctx->coarse_leny[i] = (proc_ny / s_ctx->sub_domains) + (proc_ny % s_ctx->sub_domains > i);
    s_ctx->coarse_startz[i] = s_ctx->coarse_startz[i - 1] + s_ctx->coarse_lenz[i - 1];
    s_ctx->coarse_lenz[i] = (proc_nz / s_ctx->sub_domains) + (proc_nz % s_ctx->sub_domains > i);
  }
  for (i = 0; i < s_ctx->sub_domains; ++i) {
    s_ctx->coarse_p_startx[i] = s_ctx->coarse_startx[i] - s_ctx->over_sampling >= 0 ? s_ctx->coarse_startx[i] - s_ctx->over_sampling : 0;
    s_ctx->coarse_p_lenx[i] = s_ctx->coarse_startx[i] + s_ctx->coarse_lenx[i] + s_ctx->over_sampling <= s_ctx->M ? s_ctx->coarse_startx[i] + s_ctx->coarse_lenx[i] + s_ctx->over_sampling - s_ctx->coarse_p_startx[i] : s_ctx->M - s_ctx->coarse_p_startx[i];
    s_ctx->coarse_p_starty[i] = s_ctx->coarse_starty[i] - s_ctx->over_sampling >= 0 ? s_ctx->coarse_starty[i] - s_ctx->over_sampling : 0;
    s_ctx->coarse_p_leny[i] = s_ctx->coarse_starty[i] + s_ctx->coarse_leny[i] + s_ctx->over_sampling <= s_ctx->N ? s_ctx->coarse_starty[i] + s_ctx->coarse_leny[i] + s_ctx->over_sampling - s_ctx->coarse_p_starty[i] : s_ctx->N - s_ctx->coarse_p_starty[i];
    s_ctx->coarse_p_startz[i] = s_ctx->coarse_startz[i] - s_ctx->over_sampling >= 0 ? s_ctx->coarse_startz[i] - s_ctx->over_sampling : 0;
    s_ctx->coarse_p_lenz[i] = s_ctx->coarse_startz[i] + s_ctx->coarse_lenz[i] + s_ctx->over_sampling <= s_ctx->P ? s_ctx->coarse_startz[i] + s_ctx->coarse_lenz[i] + s_ctx->over_sampling - s_ctx->coarse_p_startz[i] : s_ctx->P - s_ctx->coarse_p_startz[i];
  }

  meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  meas_face_yz = s_ctx->H_y * s_ctx->H_z;
  meas_face_zx = s_ctx->H_z * s_ctx->H_x;
  meas_face_xy = s_ctx->H_x * s_ctx->H_y;

  /*********************************************************************
    Get level-1 (overlap) block Jacobi.
  *********************************************************************/
  Mat A_i;
  Vec kappa_loc[DIM];
  PetscScalar ***arr_kappa_3d[DIM], avg_kappa_e, val_A[2][2];
  PetscInt coarse_elem_p_x, coarse_elem_p_y, coarse_elem_p_z, coarse_elem_p, row[2], col[2], coarse_elem_p_num = s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  s_ctx->ksp_lv1 = (KSP *)malloc(coarse_elem_p_num * sizeof(KSP));

  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES, kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }

  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
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
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1;
            row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1;
            col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          } else if (ex >= 1)
          // Left boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[0][ez][ey][ex];
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ex + 1 == startx + nx && ex + 1 != s_ctx->M)
          // Right boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[0][ez][ey][ex];
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }

          if (ey >= starty + 1)
          // Inner y-direction edges.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex - startx;
            row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex - startx;
            col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
            val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          } else if (ey >= 1)
          // Down boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[1][ez][ey][ex];
            val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ey + 1 == starty + ny && ey + 1 != s_ctx->N)
          // Up boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[1][ez][ey][ex];
            val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }

          if (ez >= startz + 1)
          // Inner z-direction edges.
          {
            row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex - startx;
            row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex - startx;
            col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
            val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
          } else if (ez >= 1)
          // Back boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[2][ez][ey][ex];
            val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
          if (ez + 1 == startz + nz && ez + 1 != s_ctx->P)
          // Front boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[2][ez][ey][ex];
            val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
          }
        }
    PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &(s_ctx->ksp_lv1[coarse_elem_p])));
    PetscCall(KSPSetOperators(s_ctx->ksp_lv1[coarse_elem_p], A_i, A_i));
    {
      PC pc_;
      PetscCall(KSPSetType(s_ctx->ksp_lv1[coarse_elem_p], KSPPREONLY));
      PetscCall(KSPGetPC(s_ctx->ksp_lv1[coarse_elem_p], &pc_));
      PetscCall(PCSetType(pc_, PCCHOLESKY));
      PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERCHOLMOD));
      PetscCall(PCSetOptionsPrefix(pc_, "kspl1_"));
      PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv1[coarse_elem_p], PETSC_TRUE));
      PetscCall(PCSetFromOptions(pc_));
    }
    PetscCall(KSPSetUp(s_ctx->ksp_lv1[coarse_elem_p]));
    PetscCall(MatDestroy(&A_i));
  }

  /*********************************************************************
    Get eigenvectors level-1.
  *********************************************************************/
  Mat A_i_inner, M_i;
  Vec diag_M_i;
  PetscScalar ***arr_ms_bases_loc, ***arr_M_i_3d;
  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i) {
    PetscCall(DMCreateLocalVector(s_ctx->dm, &(s_ctx->ms_bases[i])));
  }

  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
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
          arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] = 0.0;
          if (ex >= startx_ + 1) {
            row[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_ - 1;
            row[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            col[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_ - 1;
            col[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
          } else if (ex >= 1) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
          }
          if (ex + 1 != s_ctx->M) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 / arr_kappa_3d[0][ez][ey][ex + 1]);
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
          }

          if (ey >= starty_ + 1) {
            row[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_ - 1) * nx_ + ex - startx_;
            row[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            col[0] = (ez - startz_) * ny_ * nx_ + (ey - starty_ - 1) * nx_ + ex - startx_;
            col[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
            val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
          } else if (ey >= 1) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
          }
          if (ey + 1 != s_ctx->N) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 / arr_kappa_3d[1][ez][ey + 1][ex]);
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
          }

          if (ez >= startz_ + 1) {
            row[0] = (ez - startz_ - 1) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            row[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            col[0] = (ez - startz_ - 1) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            col[1] = (ez - startz_) * ny_ * nx_ + (ey - starty_) * nx_ + ex - startx_;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
            val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0], &val_A[0][0], ADD_VALUES));
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
          } else if (ez >= 1) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
          }
          if (ez + 1 != s_ctx->P) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 / arr_kappa_3d[2][ez + 1][ey][ex]);
            arr_M_i_3d[ez - startz_][ey - starty_][ex - startx_] += 2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
          }
        }
      }
    PetscCall(VecRestoreArray3d(diag_M_i, nz_, ny_, nx_, 0, 0, 0, &arr_M_i_3d));
    PetscCall(VecScale(diag_M_i, meas_elem * 0.25 * M_PI * M_PI / (nx_ * nx_ * s_ctx->H_x * s_ctx->H_x + ny_ * ny_ * s_ctx->H_y * s_ctx->H_y + nz_ * nz_ * s_ctx->H_z * s_ctx->H_z)));
    PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatDiagonalSet(M_i, diag_M_i, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(M_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(M_i, MAT_FINAL_ASSEMBLY));
    PetscCall(VecDestroy(&diag_M_i));

    EPS eps;
    PetscInt nconv;
    PetscScalar eig_val, ***arr_eig_vec;
    Vec eig_vec;

    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, A_i_inner, M_i));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetDimensions(eps, s_ctx->max_eigen_num_lv1, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));
    ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetTarget(eps, -NIL));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscCheck(nconv >= s_ctx->max_eigen_num_lv1, PETSC_COMM_WORLD, PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors! (nconv=%d, eigen_num=%d)\n", nconv, s_ctx->max_eigen_num_lv1);
    PetscBool find_max_eigen = PETSC_FALSE;
    for (j = 0; j < s_ctx->max_eigen_num_lv1; ++j) {
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      if (j == 0)
        s_ctx->eigen_min_lv1[coarse_elem_p] = eig_val;

      if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv1) {
        s_ctx->eigen_num_lv1[coarse_elem_p] = j + 1;
        s_ctx->eigen_max_lv1[coarse_elem_p] = eig_val;
        find_max_eigen = PETSC_TRUE;
      }

      if (!find_max_eigen && j == s_ctx->max_eigen_num_lv1 - 1) {
        s_ctx->eigen_num_lv1[coarse_elem_p] = s_ctx->max_eigen_num_lv1;
        s_ctx->eigen_max_lv1[coarse_elem_p] = eig_val;
      }

      PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases[j], &arr_ms_bases_loc));
      PetscCall(VecGetArray3d(eig_vec, nz_, ny_, nx_, 0, 0, 0, &arr_eig_vec));
      for (ez = startz_; ez < startz_ + nz_; ++ez)
        for (ey = starty_; ey < starty_ + ny_; ++ey)
          PetscCall(PetscArraycpy(&arr_ms_bases_loc[ez][ey][startx_], &arr_eig_vec[ez - startz_][ey - starty_][0], nx_));
      PetscCall(VecRestoreArray3d(eig_vec, nz_, ny_, nx_, 0, 0, 0, &arr_eig_vec));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases[j], &arr_ms_bases_loc));
    }
    // Do some cleaning.
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));
    PetscCall(MatDestroy(&M_i));
  }

  Vec dummy_ms_bases_glo;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, s_ctx->ms_bases[i], INSERT_VALUES, dummy_ms_bases_glo));
    PetscCall(DMGlobalToLocal(s_ctx->dm, dummy_ms_bases_glo, INSERT_VALUES, s_ctx->ms_bases[i]));
  }
  PetscCall(VecDestroy(&dummy_ms_bases_glo));

  /*********************************************************************
    Get lever-2 block Jacobi.
  *********************************************************************/
  PetscInt dof_idx[coarse_elem_p_num + 1], coarse_elem_p_col;
  PetscScalar ***arr_ms_bases_array[s_ctx->max_eigen_num_lv1];
  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases[i], &arr_ms_bases_array[i]));
  dof_idx[0] = 0;
  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p)
    dof_idx[coarse_elem_p + 1] = dof_idx[coarse_elem_p] + s_ctx->eigen_num_lv1[coarse_elem_p];
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, dof_idx[coarse_elem_p_num], dof_idx[coarse_elem_p_num], 7 * s_ctx->max_eigen_num_lv1, NULL, &A_i));
  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
    coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
    coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
    coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
    startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
    nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
    starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
    ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
    startz_ = s_ctx->coarse_startz[coarse_elem_p_z];
    nz_ = s_ctx->coarse_lenz[coarse_elem_p_z];
    for (i = 0; i < s_ctx->eigen_num_lv1[coarse_elem_p]; ++i) {
      row[0] = dof_idx[coarse_elem_p] + i;
      for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p]; ++j) {
        col[0] = dof_idx[coarse_elem_p] + j;
        val_A[0][0] = 0.0;
        for (ez = startz_; ez < startz_ + nz_; ++ez)
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              if (ex >= startx_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
                val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey][ex - 1] - arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey][ex - 1] - arr_ms_bases_array[j][ez][ey][ex]);
              } else if (ex >= 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
                val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
              if (ex + 1 == startx_ + nx_ && ex + 1 != s_ctx->M) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 / arr_kappa_3d[0][ez][ey][ex + 1]);
                val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }

              if (ey >= starty_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
                val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey - 1][ex] - arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey - 1][ex] - arr_ms_bases_array[j][ez][ey][ex]);
              } else if (ey >= 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
                val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
              if (ey + 1 == starty_ + ny_ && ey + 1 != s_ctx->N) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 / arr_kappa_3d[1][ez][ey + 1][ex]);
                val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }

              if (ez >= startz_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
                val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez - 1][ey][ex] - arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez - 1][ey][ex] - arr_ms_bases_array[j][ez][ey][ex]);
              } else if (ez >= 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
                val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
              if (ez + 1 == startz_ + nz_ && ez + 1 != s_ctx->P) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 / arr_kappa_3d[2][ez + 1][ey][ex]);
                val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
            }
        PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
      }

      if (coarse_elem_p_x != 0) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x - 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ey = starty_; ey < starty_ + ny_; ++ey) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][startx_ - 1] + 1.0 / arr_kappa_3d[0][ez][ey][startx_]);
              val_A[0][0] -= meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx_] * arr_ms_bases_array[j][ez][ey][startx_ - 1];
            }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_x != s_ctx->sub_domains - 1) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x + 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ey = starty_; ey < starty_ + ny_; ++ey) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][startx_ + nx_ - 1] + 1.0 / arr_kappa_3d[0][ez][ey][startx_ + nx_]);
              val_A[0][0] -= meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx_ + nx_ - 1] * arr_ms_bases_array[j][ez][ey][startx_ + nx_];
            }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_y != 0) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + (coarse_elem_p_y - 1) * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty_ - 1][ex] + 1.0 / arr_kappa_3d[1][ez][starty_][ex]);
              val_A[0][0] -= meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][starty_][ex] * arr_ms_bases_array[j][ez][starty_ - 1][ex];
            }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_y != s_ctx->sub_domains - 1) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + (coarse_elem_p_y + 1) * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty_ + ny_ - 1][ex] + 1.0 / arr_kappa_3d[1][ez][starty_ + ny_][ex]);
              val_A[0][0] -= meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][starty_ + ny_ - 1][ex] * arr_ms_bases_array[j][ez][starty_ + ny_][ex];
            }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_z != 0) {
        coarse_elem_p_col = (coarse_elem_p_z - 1) * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][startz_ - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz_][ey][ex]);
              val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][startz_][ey][ex] * arr_ms_bases_array[j][startz_ - 1][ey][ex];
            }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_z != s_ctx->sub_domains - 1) {
        coarse_elem_p_col = (coarse_elem_p_z + 1) * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][startz_ + nz_ - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz_ + nz_][ey][ex]);
              val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][startz_ + nz_ - 1][ey][ex] * arr_ms_bases_array[j][startz_ + nz_][ey][ex];
            }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
  PetscCall(KSPCreate(PETSC_COMM_SELF, &s_ctx->ksp_lv2));
  PetscCall(KSPSetOperators(s_ctx->ksp_lv2, A_i, A_i));
  {
    PC pc_;
    PetscCall(KSPSetType(s_ctx->ksp_lv2, KSPPREONLY));
    PetscCall(KSPGetPC(s_ctx->ksp_lv2, &pc_));
    PetscCall(PCSetType(pc_, PCCHOLESKY));
    PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERCHOLMOD));
    PetscCall(PCSetOptionsPrefix(pc_, "kspl2_"));
    PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv2, PETSC_TRUE));
    PetscCall(PCSetFromOptions(pc_));
  }
  PetscCall(KSPSetUp(s_ctx->ksp_lv2));
  PetscCall(MatDestroy(&A_i));

  /*********************************************************************
    Get eigenvectors lever-2.
  *********************************************************************/
  Vec ms_bases_tmp;
  PetscScalar ***arr_ms_bases_tmp;
  PetscCall(DMGetLocalVector(s_ctx->dm, &ms_bases_tmp));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, dof_idx[coarse_elem_p_num], dof_idx[coarse_elem_p_num], 7 * s_ctx->max_eigen_num_lv1, NULL, &A_i_inner));
  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
    coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
    coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
    coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);
    startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
    nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
    starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
    ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
    startz_ = s_ctx->coarse_startz[coarse_elem_p_z];
    nz_ = s_ctx->coarse_lenz[coarse_elem_p_z];
    for (i = 0; i < s_ctx->eigen_num_lv1[coarse_elem_p]; ++i) {
      row[0] = dof_idx[coarse_elem_p] + i;
      for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p]; ++j) {
        col[0] = dof_idx[coarse_elem_p] + j;
        val_A[0][0] = 0.0;
        for (ez = startz_; ez < startz_ + nz_; ++ez)
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              if (ex >= startx_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
                val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey][ex - 1] - arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey][ex - 1] - arr_ms_bases_array[j][ez][ey][ex]);
              } else if (ex != proc_startx) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
                val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
              if (ex + 1 == starty_ + ny_ && ex + 1 != proc_startx + proc_nx) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 / arr_kappa_3d[0][ez][ey][ex + 1]);
                val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }

              if (ey >= starty_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
                val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey - 1][ex] - arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey - 1][ex] - arr_ms_bases_array[j][ez][ey][ex]);
              } else if (ey != proc_starty) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
                val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
              if (ey + 1 == starty_ + ny_ && ey + 1 != proc_starty + proc_ny) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 / arr_kappa_3d[1][ez][ey + 1][ex]);
                val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }

              if (ez >= startz_ + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
                val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez - 1][ey][ex] - arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez - 1][ey][ex] - arr_ms_bases_array[j][ez][ey][ex]);
              } else if (ez != proc_startz) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
                val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
              if (ez + 1 == startz_ + nz_ && ez + 1 != proc_startz + proc_nz) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 / arr_kappa_3d[2][ez + 1][ey][ex]);
                val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] * arr_ms_bases_array[j][ez][ey][ex];
              }
            }
        PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
      }

      if (coarse_elem_p_x != 0) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x - 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ey = starty_; ey < starty_ + ny_; ++ey) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][startx_ - 1] + 1.0 / arr_kappa_3d[0][ez][ey][startx_]);
              val_A[0][0] -= meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx_] * arr_ms_bases_array[j][ez][ey][startx_ - 1];
            }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_x != s_ctx->sub_domains - 1) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x + 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ey = starty_; ey < starty_ + ny_; ++ey) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][startx_ + nx_ - 1] + 1.0 / arr_kappa_3d[0][ez][ey][startx_ + nx_]);
              val_A[0][0] -= meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx_ + nx_ - 1] * arr_ms_bases_array[j][ez][ey][startx_ + nx_];
            }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_y != 0) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + (coarse_elem_p_y - 1) * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty_ - 1][ex] + 1.0 / arr_kappa_3d[1][ez][starty_][ex]);
              val_A[0][0] -= meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][starty_][ex] * arr_ms_bases_array[j][ez][starty_ - 1][ex];
            }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_y != s_ctx->sub_domains - 1) {
        coarse_elem_p_col = coarse_elem_p_z * s_ctx->sub_domains * s_ctx->sub_domains + (coarse_elem_p_y + 1) * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ez = startz_; ez < startz_ + nz_; ++ez)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty_ + ny_ - 1][ex] + 1.0 / arr_kappa_3d[1][ez][starty_ + ny_][ex]);
              val_A[0][0] -= meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][starty_ + ny_ - 1][ex] * arr_ms_bases_array[j][ez][starty_ + ny_][ex];
            }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_z != 0) {
        coarse_elem_p_col = (coarse_elem_p_z - 1) * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][startz_ - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz_][ey][ex]);
              val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][startz_][ey][ex] * arr_ms_bases_array[j][startz_ - 1][ey][ex];
            }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }

      if (coarse_elem_p_z != s_ctx->sub_domains - 1) {
        coarse_elem_p_col = (coarse_elem_p_z + 1) * s_ctx->sub_domains * s_ctx->sub_domains + coarse_elem_p_y * s_ctx->sub_domains + coarse_elem_p_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
          col[0] = dof_idx[coarse_elem_p_col] + j;
          val_A[0][0] = 0.0;
          for (ey = starty_; ey < starty_ + ny_; ++ey)
            for (ex = startx_; ex < startx_ + nx_; ++ex) {
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][startz_ + nz_ - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz_ + nz_][ey][ex]);
              val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e * arr_ms_bases_array[i][startz_ + nz_ - 1][ey][ex] * arr_ms_bases_array[j][startz_ + nz_][ey][ex];
            }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0], &val_A[0][0], ADD_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));

  s_ctx->ms_bases_c = (Vec *)malloc(s_ctx->max_eigen_num_lv2 * sizeof(Vec));
  for (i = 0; i < s_ctx->max_eigen_num_lv2; ++i)
    PetscCall(DMCreateLocalVector(s_ctx->dm, &s_ctx->ms_bases_c[i]));

  EPS eps;
  PetscInt nconv;
  PetscScalar eig_val, *arr_eig_vec, ****arr_ms_bases_c;
  Vec eig_vec;

  PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
  PetscCall(EPSSetOperators(eps, A_i_inner, NULL));
  PetscCall(EPSSetProblemType(eps, EPS_HEP));
  PetscCall(EPSSetDimensions(eps, s_ctx->max_eigen_num_lv2, PETSC_DEFAULT, PETSC_DEFAULT));
  ST st;
  PetscCall(EPSGetST(eps, &st));
  PetscCall(STSetType(st, STSINVERT));
  PetscCall(EPSSetTarget(eps, -NIL));
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetConverged(eps, &nconv));
  PetscCheck(nconv >= s_ctx->max_eigen_num_lv2, PETSC_COMM_WORLD, PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors for level-2! (nconv=%d, eigen_num=%d)\n", nconv, s_ctx->max_eigen_num_lv2);
  PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));
  PetscBool find_max_eigen = PETSC_FALSE;
  for (j = 0; j < s_ctx->max_eigen_num_lv2; ++j) {
    PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
    if (!find_max_eigen && j == 0)
      s_ctx->eigen_min_lv2 = eig_val;

    if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv2) {
      s_ctx->eigen_num_lv2 = j + 1;
      s_ctx->eigen_max_lv2 = eig_val;
      find_max_eigen = PETSC_TRUE;
    }

    if (j == s_ctx->max_eigen_num_lv2 - 1) {
      s_ctx->eigen_num_lv2 = s_ctx->max_eigen_num_lv1;
      s_ctx->eigen_max_lv2 = eig_val;
    }
    PetscCall(VecGetArray(eig_vec, &arr_eig_vec));
    // Construct coarse-coarse bases in the original P space.
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases_c[j], &arr_ms_bases_c));

    for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
      coarse_elem_p_x = coarse_elem_p % s_ctx->sub_domains;
      coarse_elem_p_y = (coarse_elem_p / s_ctx->sub_domains) % s_ctx->sub_domains;
      coarse_elem_p_z = coarse_elem_p / (s_ctx->sub_domains * s_ctx->sub_domains);

      // Insert data into ms_bases_c.
      PetscCall(VecZeroEntries(ms_bases_tmp));
      PetscCall(VecMAXPY(ms_bases_tmp, s_ctx->eigen_num_lv1[coarse_elem_p], &arr_eig_vec[dof_idx[coarse_elem_p]], &s_ctx->ms_bases[0]));
      PetscCall(DMDAVecGetArray(s_ctx->dm, ms_bases_tmp, &arr_ms_bases_tmp));
      startx_ = s_ctx->coarse_startx[coarse_elem_p_x];
      nx_ = s_ctx->coarse_lenx[coarse_elem_p_x];
      starty_ = s_ctx->coarse_starty[coarse_elem_p_y];
      ny_ = s_ctx->coarse_leny[coarse_elem_p_y];
      startz_ = s_ctx->coarse_startz[coarse_elem_p_z];
      nz_ = s_ctx->coarse_lenz[coarse_elem_p_z];
      for (ez = startz_; ez < startz_ + nz_; ++ez)
        for (ey = starty_; ey < starty_ + ny_; ++ey)
          PetscCall(PetscArraycpy(&arr_ms_bases_c[ez][ey][ex], &arr_ms_bases_tmp[ez][ey][ex], nx_));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm, ms_bases_tmp, &arr_ms_bases_tmp));
    }
    PetscCall(VecRestoreArray(eig_vec, &arr_eig_vec));
  }
  dof_idx[coarse_elem_p_num + 1] = s_ctx->eigen_num_lv2;
  // Do some cleaning.
  PetscCall(VecDestroy(&eig_vec));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A_i_inner));

  PetscCall(DMCreateGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < s_ctx->max_eigen_num_lv2; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, s_ctx->ms_bases_c[i], INSERT_VALUES, dummy_ms_bases_glo));
    PetscCall(DMGlobalToLocal(s_ctx->dm, dummy_ms_bases_glo, INSERT_VALUES, s_ctx->ms_bases_c[i]));
  }
  PetscCall(VecDestroy(&dummy_ms_bases_glo));

  /*********************************************************************
    Get level-3.
  *********************************************************************/
  Mat A_cc;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, s_ctx->eigen_num_lv2, s_ctx->eigen_num_lv2, PETSC_DETERMINE, PETSC_DETERMINE, s_ctx->eigen_num_lv2, NULL, 6 * s_ctx->max_eigen_num_lv2, NULL, &A_cc));
  PetscInt A_cc_range[NEIGH + 1][2];
  PetscCall(MatGetOwnershipRange(A_cc, &A_cc_range[0][0], &A_cc_range[0][1]));
  const PetscMPIInt *ng_ranks;
  PetscCall(DMDAGetNeighbors(s_ctx->dm, &ng_ranks));
  // Send dof_idx first.
  if (proc_startx != 0)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[12], ng_ranks[13], PETSC_COMM_WORLD)); // Left.
  if (proc_startx + proc_nx != s_ctx->M)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[14], ng_ranks[13], PETSC_COMM_WORLD)); // Right.
  if (proc_starty != 0)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[10], ng_ranks[13], PETSC_COMM_WORLD)); // Down.
  if (proc_starty + proc_ny != s_ctx->N)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[16], ng_ranks[13], PETSC_COMM_WORLD)); // Up.
  if (proc_startz != 0)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[4], ng_ranks[13], PETSC_COMM_WORLD)); // Back.
  if (proc_startz + proc_nz != s_ctx->P)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[26], ng_ranks[13], PETSC_COMM_WORLD)); // Front.
  // Receive dof_idx then.
  MPI_Status ista;
  if (proc_startx != 0)
    PetscCallMPI(MPI_Recv(&A_cc_range[1][0], 2, MPI_INT, ng_ranks[12], ng_ranks[12], PETSC_COMM_WORLD, &ista)); // Left.
  if (proc_startx + proc_nx != s_ctx->M)
    PetscCallMPI(MPI_Recv(&A_cc_range[2][0], 2, MPI_INT, ng_ranks[14], ng_ranks[14], PETSC_COMM_WORLD, &ista)); // Right.
  if (proc_starty != 0)
    PetscCallMPI(MPI_Recv(&A_cc_range[3][0], 2, MPI_INT, ng_ranks[10], ng_ranks[10], PETSC_COMM_WORLD, &ista)); // Down.
  if (proc_starty + proc_ny != s_ctx->N)
    PetscCallMPI(MPI_Recv(&A_cc_range[4][0], 2, MPI_INT, ng_ranks[16], ng_ranks[16], PETSC_COMM_WORLD, &ista)); // Up.
  if (proc_startz != 0)
    PetscCallMPI(MPI_Recv(&A_cc_range[5][0], 2, MPI_INT, ng_ranks[4], ng_ranks[4], PETSC_COMM_WORLD, &ista)); // Back.
  if (proc_startz + proc_nz != s_ctx->P)
    PetscCallMPI(MPI_Recv(&A_cc_range[6][0], 2, MPI_INT, ng_ranks[22], ng_ranks[22], PETSC_COMM_WORLD, &ista)); // Front.

  PetscFunctionReturn(0);
}

PetscErrorCode PC_final_default(PCCtx **s_ctx_) {
  PetscFunctionBeginUser;
  PCCtx *s_ctx = *s_ctx_;

  //   PetscCall(PetscFree(s_ctx->A_i));
  PetscCall(PetscFree(s_ctx->ms_bases));
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