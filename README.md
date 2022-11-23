### Configure PETSc

#### Debug mode in the local machine with system default compilers
./configure PETSC_ARCH=linux-gcc-debug --with-debugging=1 \
--download-fblaslapack \
--download-mpich  \
--download-slepc \
--download-cmake \
--download-scalapack \
--download-mumps \
--download-suitesparse \
--download-hdf5


#### Optimized mode in the local machine with oneAPI MKL
Set environment variables first.

source /opt/intel/oneapi/setvars.sh

Intel oneAPI does not have mpi-wrappered icx/icpx. Fail to use icx/icpx to compile SCALAPACK.
Read [Quick Reference Guide to Optimization with Intel® C++ and Fortran Compilers].

./configure PETSC_ARCH=linux-oneAPI-opt --with-debugging=0 \
--CFLAGS='-diag-disable=10441 -O3 -qopenmp -qmkl' \
--FFLAGS='-O3 -qopenmp -qmkl' \
--CXXFLAGS='-diag-disable=10441 -O3 -qopenmp -qmkl' \
--with-cc=mpiicc --with-fc=mpiifort --with-cxx=mpiicpc \
--download-scalapack \
--download-slepc \
--download-cmake \
--download-mumps \
--download-suitesparse \
--download-hdf5

Check the current configuration file.

cat linux-oneAPI-opt/lib/petsc/conf/reconfigure-linux-oneAPI-debug.py 


### Generate compile_commands.json with Bear
I tried Bear on a simple ``HelloWorld'' project, it failed (output a empty json file) when using icc.
A remedy is using PETSC_ARCH=linux-gcc-debug when writing code, and installing Bear with apt-get.
Use the following command to generate compile_commands.json, which is needed by the Clangd plugin of VS Code. 

Bear -- make [a_test_project]




