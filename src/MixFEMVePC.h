#ifndef YCQ_MIXFEM_VE_PC_H
#define YCQ_MIXFEM_VE_PC_H

#include "petsclog.h"
#include "petscsystypes.h"
static char help[] = "Solve a heterogeneous coefficient second order ellipic PDE with the RT0-Q0 mixed finite element method and the velocity elimination technique, while the final system is preconditioned.";

#include "util.h"
#include <petscdmda.h>
#include <petscksp.h>
#include <petsctime.h>
#include <slepceps.h>

#define MAX_DIM 3

typedef struct MixFEMVePCCtx {
  DM dm_os, dm_coarse_sp;
  //   Mat A_0, *A_i;          // free(A_i).
  Vec v_kappa, *ms_bases; // free(ms_bases).
  KSP ksp_0, *ksp_i;      // free(ksp_i)
  PetscBool A_0_off, A_i_off;
  PetscScalar *eigen_max, *eigen_min, robin_alpha;
  PetscInt *coarse_startx, *coarse_lenx, *coarse_starty, *coarse_leny, *coarse_startz, *coarse_lenz;
  PetscInt *coarse_p_startx, *coarse_p_lenx, *coarse_p_starty, *coarse_p_leny, *coarse_p_startz, *coarse_p_lenz, coarse_elem_p_num;
  PetscInt coarse_total_nx, coarse_total_ny, coarse_total_nz;
  PetscInt over_sampling, sub_domains, eigen_num, icc_level;
  int M, N, P;
  double H_x, H_y, H_z;
  ModelContext m_ctx;
  PetscLogStage su0, su1, su2, av0, av1;
  PetscLogDouble su[3], av[2];
  //   PetscLogEvent su0, su1, su2, av0, av1;
  //   PetscClassId su, av;
} PCCtx;

PetscErrorCode PC_init(PCCtx **s_ctx, ModelContext m_ctx, int M, int N, int P, int o_s, int s_d, int e_n, int i_l, PetscBool turn_off_A_0, PetscBool turn_off_A_i, PetscScalar robin_alpha);

PetscErrorCode PC_final_default(PCCtx **s_ctx_);

PetscErrorCode PC_final(PCCtx **s_ctx_);

PetscErrorCode PC_setup(PC pc);

PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y);

PetscErrorCode PC_apply_vec_hybrid(PC pc, Vec x, Vec y);

PetscErrorCode PC_write_kappa(PCCtx *s_ctx);

PetscErrorCode PC_create_system(PCCtx *s_ctx, Vec v_source, Mat *A, Vec *rhs);

#endif