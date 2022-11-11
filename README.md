### Configure PETSc

#### Debug mode in the local machine
./configure PETSC_ARCH=linux-c-debug \
--download-fblaslapack --download-mpich --with-debugging=1 \
--download-slepc \
--download-cmake --download-scalapack --download-mumps \
--download-suitesparse --download-hdf5