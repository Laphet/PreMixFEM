#ifndef YCQ_MIXFEM_3L3D_H
#define YCQ_MIXFEM_3L3D_H

#include <mpi.h>
#include <petscksp.h>

#define DIM 3
#define NEIGH 6
#define NIL 1.0e-12
#define MAX_LOG_STATES 7
#define STAGE_SU_LV1 0
#define STAGE_SU_LV2 1
#define STAGE_SU_LV3 2
#define STAGE_AV_LV1 3
#define STAGE_AV_LV2 4
#define STAGE_AV_LV3 5
#define STAGE_AV 6

typedef struct preconditioner_context {
  DM dm;
  Vec kappa[DIM], *ms_bases_c, *ms_bases_cc;
  KSP *ksp_lv1, ksp_lv2, ksp_lv3;
  PetscInt *coarse_startx, *coarse_lenx, *coarse_starty, *coarse_leny, *coarse_startz, *coarse_lenz;
  PetscInt sub_domains;
  PetscInt max_eigen_num_lv1, max_eigen_num_lv1_upd, *eigen_num_lv1;
  PetscInt max_eigen_num_lv2, max_eigen_num_lv2_upd, eigen_num_lv2;
  PetscInt M, N, P;
  PetscScalar H_x, H_y, H_z, L, W, H;
  PetscScalar *eigen_max_lv1, *eigen_min_lv1, eigen_bd_lv1, eigen_max_lv2, eigen_min_lv2, eigen_bd_lv2;
  PetscScalar t_stages[MAX_LOG_STATES];
  PetscBool use_W_cycle, no_shift_A_cc;
} PCCtx;

PetscErrorCode PC_init(PCCtx *s_ctx, PetscScalar *dom, PetscInt *mesh, PetscScalar *fl_args, PetscInt *int_args, PetscBool *b_args);
/*
    dom[0], the length; dom[1], the width; dom[2], the height.
    mesh[0], partions in x-direction; mesh[1], partions in y-direction; mesh[2], partions in z-direction.
    fl_args[0] > 0, the threshold of eigenvalues level1.
    fl_args[1] > 0, the threshold of eigenvalues level2.
    int_args[1], the number of subdomains in each direction.
    int_args[2], the number of eigenvectors solved level1.
    int_args[3], the number of eigenvectors solved level2.
    b_args[0], use W-cycle instead of default V-cycle.
*/

PetscErrorCode PC_print_info(PCCtx *s_ctx);

PetscErrorCode _PC_setup(PCCtx *s_ctx);

PetscErrorCode PC_setup(PC pc);

PetscErrorCode PC_create_A(PCCtx *s_ctx, Mat *A);

PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y);

PetscErrorCode PC_get_range(const void *sendbuff, void *recvbuff, MPI_Datatype datatype);

PetscErrorCode PC_print_stat(PCCtx *s_ctx);

PetscErrorCode PC_final_default(PCCtx *s_ctx);

PetscErrorCode PC_final(PCCtx *s_ctx);

#endif