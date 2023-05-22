#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#define ROOT_PATH "../"
#define MAX_FILENAME_LEN 72
#define DIM 3
#define PORO_MIN 0.05

PetscErrorCode _load_into_dm_vec(Vec load_from_h5_file, Vec load_into_dm_vec, DM dm) {
  PetscFunctionBeginUser;

  PetscInt startx, starty, startz, nx, ny, nz, ey, ez, M, N, P, size;
  PetscScalar ***arr_load_from_h5_file, ***load_into_dm_vec_arr;

  PetscCall(DMDAGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAGetInfo(dm, NULL, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(VecGetSize(load_from_h5_file, &size));

  PetscCheck(size == M * N * P, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Something wrong! the vector loaded from the h5 file with the size=%d while we need size=%d.\n", size, M * N * P);
  PetscCall(VecGetArray3dRead(load_from_h5_file, P, N, M, 0, 0, 0, &arr_load_from_h5_file));

  PetscCall(DMDAVecGetArray(dm, load_into_dm_vec, &load_into_dm_vec_arr));
  for (ez = startz; ez < startz + nz; ++ez)
    for (ey = starty; ey < starty + ny; ++ey) {
      PetscCall(PetscArraycpy(&load_into_dm_vec_arr[ez][ey][startx], &arr_load_from_h5_file[ez][ey][startx], nx));
    }

  PetscCall(VecRestoreArray3dRead(load_from_h5_file, P, N, M, 0, 0, 0, &arr_load_from_h5_file));

  PetscCall(DMDAVecRestoreArray(dm, load_into_dm_vec, &load_into_dm_vec_arr));

  PetscFunctionReturn(0);
}

PetscErrorCode save_vec_into_vtr(Vec v, const char *vtr_name) {
  PetscFunctionBeginUser;

  char file_name[MAX_FILENAME_LEN];
  PetscViewer vtr_viewer;

  sprintf(file_name, "%sdata/spe10_output/%s", ROOT_PATH, vtr_name);
  PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, file_name, FILE_MODE_WRITE, &vtr_viewer));
  PetscCall(VecView(v, vtr_viewer));
  PetscCall(PetscViewerDestroy(&vtr_viewer));

  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));
  PetscInt M = 60, N = 220, P = 85;
  DM dm;
  Vec load_from_h5_file, perm[DIM], poro;
  PetscViewer h5_viewer;
  char file_name[MAX_FILENAME_LEN];

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));

  PetscCall(DMCreateGlobalVector(dm, &perm[0]));
  PetscCall(PetscObjectSetName((PetscObject)perm[0], "perm_x"));
  PetscCall(DMCreateGlobalVector(dm, &perm[1]));
  PetscCall(PetscObjectSetName((PetscObject)perm[1], "perm_y"));
  PetscCall(DMCreateGlobalVector(dm, &perm[2]));
  PetscCall(PetscObjectSetName((PetscObject)perm[2], "perm_z"));
  PetscCall(DMCreateGlobalVector(dm, &poro));
  PetscCall(PetscObjectSetName((PetscObject)poro, "poro"));

  sprintf(file_name, "%s%s", ROOT_PATH, "data/spe10/SPE10_rock.h5");
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, file_name, FILE_MODE_READ, &h5_viewer));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, M * N * P, &load_from_h5_file));

  PetscCall(PetscObjectSetName((PetscObject)load_from_h5_file, "perm_x"));
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(VecScale(load_from_h5_file, 10.7639)); // 1 [m^2] = 10.7639 [ft^2].
  PetscCall(_load_into_dm_vec(load_from_h5_file, perm[0], dm));

  PetscCall(PetscObjectSetName((PetscObject)load_from_h5_file, "perm_y"));
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(VecScale(load_from_h5_file, 10.7639)); // 1 [m^2] = 10.7639 [ft^2].
  PetscCall(_load_into_dm_vec(load_from_h5_file, perm[1], dm));

  PetscCall(PetscObjectSetName((PetscObject)load_from_h5_file, "perm_z"));
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(VecScale(load_from_h5_file, 10.7639)); // 1 [m^2] = 10.7639 [ft^2].
  PetscCall(_load_into_dm_vec(load_from_h5_file, perm[2], dm));

  PetscObjectSetName((PetscObject)load_from_h5_file, "poro");
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(_load_into_dm_vec(load_from_h5_file, poro, dm));

  PetscCall(PetscViewerDestroy(&h5_viewer));
  PetscCall(VecDestroy(&load_from_h5_file));

  PetscCall(save_vec_into_vtr(perm[0], "perm_x.vtr"));
  PetscCall(save_vec_into_vtr(perm[1], "perm_y.vtr"));
  PetscCall(save_vec_into_vtr(perm[2], "perm_z.vtr"));
  PetscCall(save_vec_into_vtr(poro, "poro.vtr"));

  Vec ratio;
  PetscCall(VecDuplicate(poro, &ratio));
  PetscCall(PetscObjectSetName((PetscObject)ratio, "perm_x,y / perm_z"));
  PetscCall(VecPointwiseDivide(ratio, perm[0], perm[2]));
  PetscCall(save_vec_into_vtr(ratio, "ratio.vtr"));
  PetscCall(VecDestroy(&ratio));

  PetscScalar ***arr_poro;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex, ey, ez;
  PetscCall(DMDAGetCorners(dm, &proc_startx, &proc_starty, &proc_startz, &proc_nx, &proc_ny, &proc_nz));
  PetscCall(DMDAVecGetArray(dm, poro, &arr_poro));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        if (arr_poro[ez][ey][ex] < PORO_MIN)
          arr_poro[ez][ey][ex] = PORO_MIN;
      }
  PetscCall(DMDAVecRestoreArray(dm, poro, &arr_poro));
  PetscCall(PetscObjectSetName((PetscObject)poro, "poro_mod"));
  PetscCall(save_vec_into_vtr(poro, "poro_mod.vtr"));

  PetscCall(VecDestroy(&perm[0]));
  PetscCall(VecDestroy(&perm[1]));
  PetscCall(VecDestroy(&perm[2]));
  PetscCall(DMCreateGlobalVector(dm, &poro));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());

  return 0;
}