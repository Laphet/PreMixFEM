#include "PreMixFEM_3D.h"
#include "petscviewerhdf5.h"
#include <petscdmda.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <slepceps.h>

#define SAMPLE_LEN 256

PetscErrorCode create_fracture_kappa(PCCtx *s_ctx, PetscInt cr, PetscScalar ***data) {
  PetscFunctionBeginUser;
  PetscInt startx, nx, ex, ex_r, starty, ny, ey, ey_r, startz, nz, ez, ez_r, i;
  PetscCall(DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscScalar ***arr_kappa_array[DIM];
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &s_ctx->kappa[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->kappa[i], &arr_kappa_array[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    ez_r = ez % SAMPLE_LEN;
    for (ey = starty; ey < starty + ny; ++ey) {
      ey_r = ey % SAMPLE_LEN;
      for (ex = startx; ex < startx + nx; ++ex) {
        ex_r = ex % SAMPLE_LEN;
        if (data[ez_r][ey_r][ex_r] <= 0.5) {
          arr_kappa_array[0][ez][ey][ex] = PetscPowInt(10.0, cr);
          arr_kappa_array[1][ez][ey][ex] = PetscPowInt(10.0, cr);
          arr_kappa_array[2][ez][ey][ex] = PetscPowInt(10.0, cr);
        } else {
          arr_kappa_array[0][ez][ey][ex] = 1.0;
          arr_kappa_array[1][ez][ey][ex] = 1.0;
          arr_kappa_array[2][ez][ey][ex] = 1.0;
        }
      }
    }
  }

  for (i = 0; i < DIM; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->kappa[i], &arr_kappa_array[i]));

  PetscFunctionReturn(0);
}

PetscErrorCode create_well_source_XxY_rhs(PCCtx *s_ctx, Vec *rhs) {
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, rhs));
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz;
  PetscScalar ***arr_source_3d, meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  PetscCall(DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, *rhs, &arr_source_3d));
  for (ez = startz; ez < startz + nz; ++ez)
    for (ey = starty; ey < starty + ny; ++ey)
      for (ex = startx; ex < startx + nx; ++ex) {
        /* Change the source term values here. */
        if ((ex == 0 || ex == s_ctx->M - 1) && (ey == 0 || ey == s_ctx->N - 1))
          arr_source_3d[ez][ey][ex] = 1.0e+3 * meas_elem;
        else if ((ex == s_ctx->M / 2 - 1 || ex == s_ctx->M / 2) && (ey == s_ctx->N / 2 - 1 || ey == s_ctx->N / 2))
          arr_source_3d[ez][ey][ex] = -1.0e+3 * meas_elem;
        else
          arr_source_3d[ez][ey][ex] = 0.0;
      }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, *rhs, &arr_source_3d));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, (char *)0, "This is a code for testing adaptivity on the number of eigenfunctions!\n"));
  PetscInt mesh[3] = {512, 512, 512}, cr = 0, i;
  PetscBool is_petsc_default = PETSC_FALSE;
  PetscScalar dom[3] = {1.0, 1.0, 1.0}, norm_rhs;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-size", &mesh[0], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-size", &mesh[1], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-size", &mesh[2], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-cr", &cr, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &is_petsc_default));
  PCCtx s_ctx;
  PetscCall(PC_init(&s_ctx, &dom[0], &mesh[0]));
  PetscCall(PC_print_info(&s_ctx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Use contrast config=%d.\n", cr));

  PetscLogDouble main_stage[2] = {0.0, 0.0}, time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Vec rhs, u, r, frac_kappa;
  Mat A;
  PetscViewer reader;
  PetscScalar ***arr_frac_kappa;

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, "data/fracture256/fracture256_4.h5", FILE_MODE_READ, &reader));
  PetscCall(VecCreate(PETSC_COMM_SELF, &frac_kappa));
  PetscCall(VecSetType(frac_kappa, VECSEQ));
  PetscObjectSetName((PetscObject)frac_kappa, "frac");
  PetscCall(VecLoad(frac_kappa, reader));
  PetscCall(PetscViewerDestroy(&reader));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Successfully load the data!\n"));

  PetscCall(VecGetArray3d(frac_kappa, SAMPLE_LEN, SAMPLE_LEN, SAMPLE_LEN, 0, 0, 0, &arr_frac_kappa));
  PetscCall(create_fracture_kappa(&s_ctx, cr, arr_frac_kappa));
  PetscCall(VecRestoreArray3d(frac_kappa, SAMPLE_LEN, SAMPLE_LEN, SAMPLE_LEN, 0, 0, 0, &arr_frac_kappa));
  PetscCall(VecDestroy(&frac_kappa));
  PetscCall(PC_create_A(&s_ctx, &A));
  PetscCall(create_well_source_XxY_rhs(&s_ctx, &rhs));
  PetscCall(VecNormalize(rhs, &norm_rhs));

  PetscCall(PetscTimeSubtract(&time_tmp));
  main_stage[0] -= time_tmp;

  // Solve the system.
  PetscCall(PetscTime(&time_tmp));
  // PetscCall(_PC_setup(&s_ctx));
  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  if (!is_petsc_default) {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, &s_ctx));
    PetscCall(PCShellSetSetUp(pc, PC_setup));
    PetscCall(PCShellSetApply(pc, PC_apply_vec));
    PetscCall(PCShellSetName(pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
  }

  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetFromOptions(ksp));
  // Default is KSP_NORM_PRECONDITIONED, which is not the real residual norm.
  PetscCall(KSPSetUp(ksp));
  PetscCall(VecDuplicate(rhs, &u));
  PetscCall(KSPSolve(ksp, rhs, u));

  PetscCall(PetscTimeSubtract(&time_tmp));
  main_stage[1] -= time_tmp;

  PetscInt iter_count;
  PetscScalar residual;
  PetscCall(VecDuplicate(u, &r));
  PetscCall(MatMult(A, u, r));
  PetscCall(VecAXPY(r, -1.0, rhs));
  PetscCall(VecNorm(r, NORM_2, &residual));
  PetscCall(VecNorm(rhs, NORM_2, &norm_rhs));
  PetscCall(KSPGetIterationNumber(ksp, &iter_count));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The iteration number=%d.\n", iter_count));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The absolute residual in L2-norm=%.5f, the rhs in L2-norm=%.5f.\n", residual, norm_rhs));
  PetscCall(KSPConvergedReasonView(ksp, 0));

  if (!is_petsc_default)
    PetscCall(PC_print_stat(&s_ctx));

  PetscLogDouble main_stage_range[2][2];
  PetscCall(PC_get_range(&main_stage[0], &main_stage_range[0][0], MPI_DOUBLE));
  PetscCall(PC_get_range(&main_stage[1], &main_stage_range[1][0], MPI_DOUBLE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Pre=[%.5f, %.5f], Sol=[%.5f, %.5f].\n", main_stage_range[0][0], main_stage_range[0][1], main_stage_range[1][0], main_stage_range[1][1]));

  // ----------------
  // Cleaning.
  // ----------------
  PetscCall(VecDestroy(&r));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&rhs));
  for (i = 0; i < DIM; ++i)
    PetscCall(VecDestroy(&s_ctx.kappa[i]));
  PetscCall(MatDestroy(&A));
  if (!is_petsc_default) {
    PetscCall(PC_final(&s_ctx));
  } else {
    PetscCall(PC_final_default(&s_ctx));
  }

  PetscCall(SlepcFinalize());
  return 0;
}