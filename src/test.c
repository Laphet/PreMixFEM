#include "PreMixFEM_3D.h"
#include "mpi.h"
#include "petscdm.h"
#include "petscerror.h"
#include "petscoptions.h"
#include "petscsystypes.h"
#include "petscvec.h"

#define MAX_ARGS 24

PetscErrorCode create_Laplace_kappa(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &s_ctx->kappa[0]));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &s_ctx->kappa[1]));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &s_ctx->kappa[2]));
  PetscCall(VecSet(s_ctx->kappa[0], 1.0));
  PetscCall(VecSet(s_ctx->kappa[1], 1.0));
  PetscCall(VecSet(s_ctx->kappa[2], 1.0));
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
          arr_source_3d[ez][ey][ex] = 0.0 * meas_elem;
      }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, *rhs, &arr_source_3d));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, (char *)0, "This is a test code for the Laplace operator with a homogeneous Neumann BC!\n"));
  PetscInt mesh[3] = {8, 8, 8}, int_args[MAX_ARGS];
  PetscBool is_petsc_default = PETSC_FALSE, b_args[MAX_ARGS];
  PetscScalar dom[3] = {1.0, 1.0, 1.0}, fl_args[MAX_ARGS];
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &mesh[0], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &mesh[1], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-P", &mesh[2], NULL));
  int_args[0] = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-os", &int_args[0], NULL));
  int_args[1] = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-sd", &int_args[1], NULL));
  int_args[2] = 3;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-en_lv1", &int_args[2], NULL));
  int_args[3] = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-en_lv2", &int_args[3], NULL));
  fl_args[0] = -1.0;
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-eb_lv1", &fl_args[0], NULL));
  fl_args[1] = -1.0;
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-eb_lv2", &fl_args[1], NULL));
  //   PetscCall(PetscOptionsGetInt(NULL, NULL, "-st", &st, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &is_petsc_default));

  PCCtx s_ctx;
  PetscCall(PC_init(&s_ctx, &dom[0], &mesh[0], &fl_args[0], &int_args[0], &b_args[0]));
  PetscCall(PC_print_info(&s_ctx));
  // Build the system.
  PetscLogDouble main_stage[2] = {0.0, 0.0}, time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Vec rhs, u, r;
  Mat A;
  PetscCall(create_Laplace_kappa(&s_ctx));
  PetscCall(PC_create_A(&s_ctx, &A));
  PetscCall(create_well_source_XxY_rhs(&s_ctx, &rhs));

  PetscCall(PetscTimeSubtract(&time_tmp));
  main_stage[0] -= time_tmp;

  // Solve the system.
  PetscCall(PetscTime(&time_tmp));
  PetscCall(_PC_setup(&s_ctx));
  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  if (!is_petsc_default) {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, &s_ctx));
    // PetscCall(PCShellSetSetUp(pc, PC_setup));
    PetscCall(PCShellSetApply(pc, PC_apply_vec));
    PetscCall(PCShellSetName(pc, "3levels-ASM-via-GMsFEM-with-velocity-elimination"));
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
  PetscCall(KSPGetIterationNumber(ksp, &iter_count));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The iteration number=%d.\n", iter_count));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The absolute residual in L2-norm=%.5f.\n", residual));
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
  PetscCall(MatDestroy(&A));
  if (!is_petsc_default) {
    PetscCall(PC_final(&s_ctx));
  } else {
    PetscCall(PC_final_default(&s_ctx));
  }
  PetscCall(SlepcFinalize());
  return 0;
}