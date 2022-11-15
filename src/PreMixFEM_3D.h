#ifndef YCQ_MIXFEM_3L3D_H
#define YCQ_MIXFEM_3L3D_H

#include "petscsystypes.h"
#include <petscdmda.h>
#include <petscksp.h>
#include <slepceps.h>

#define DIM 3
#define NEIGH 6
#define NIL 1.0e-12

typedef struct PC_Context {
  DM dm;
  Vec kappa[DIM], *ms_bases, *ms_bases_c;
  KSP *ksp_lv1, ksp_lv2, ksp_lv3;
  PetscInt *coarse_startx, *coarse_lenx, *coarse_starty, *coarse_leny, *coarse_startz, *coarse_lenz;
  PetscInt *coarse_p_startx, *coarse_p_lenx, *coarse_p_starty, *coarse_p_leny, *coarse_p_startz, *coarse_p_lenz;
  PetscInt over_sampling, sub_domains, max_eigen_num_lv1, max_eigen_num_lv1_upd, *eigen_num_lv1, max_eigen_num_lv2, max_eigen_num_lv2_upd, eigen_num_lv2, M, N, P;
  PetscScalar H_x, H_y, H_z, L, W, H, *eigen_max_lv1, *eigen_min_lv1, eigen_bd_lv1, eigen_max_lv2, eigen_min_lv2, eigen_bd_lv2;
} PCCtx;

PetscErrorCode PC_init(PCCtx **init_ctx, PetscScalar *dom, PetscInt *mesh, PetscScalar *fl_args, PetscInt *int_args, PetscBool *b_args);
/*
    dom[0], the length; dom[1], the width; dom[2], the height.
    mesh[0], partions in x-direction; mesh[1], partions in y-direction; mesh[2], partions in z-direction.
    fl_args[0] > 0, the threshold of eigenvalues level1.
    fl_args[1] > 0, the threshold of eigenvalues level2.
    int_args[0], the number of oversampling layers.
    int_args[1], the number of subdomains in each direction.
    int_args[2], the number of eigenvectors solved level1.
    int_args[3], the number of eigenvectors solved level2.
*/

PetscErrorCode PC_setup(PC pc);

PetscErrorCode PC_setup_lite(PC pc);

PetscErrorCode PC_final_default(PCCtx **s_ctx_);

PetscErrorCode PC_final(PCCtx **s_ctx_);

#endif