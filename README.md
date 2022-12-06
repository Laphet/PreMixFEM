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

#### Optimized mode in the LSSC_IV machine with oneAPI MKL MPI
In the LSSC-IV machine (valid in Dec. 2022), try this configuration. 
CPUs are [Intel Xeon Gold 6140] and their instruction sets contain [Intel® SSE4.2, Intel® AVX, Intel® AVX2, Intel® AVX-512].
Note if set '-Ofast' for ifort, the compiling of ScaLAPACK is extremely slow.
The net connection is blocked, several packages need to be downloaded.

module load mpi/oneapimpi-oneapi-2021.1.1

module load apps/mkl-oneapi-2021

./configure PETSC_ARCH=linux-oneAPI-opt --with-debugging=0 \
--CFLAGS='-Ofast -qopenmp -xhost' \
--FFLAGS='-O3 -qopenmp -xhost' \
--CXXFLAGS='-Ofast -qopenmp -xhost' \
--with-blaslapack-dir=${MKLROOT} \
--with-mpi-dir=${I_MPI_ROOT} \
--download-scalapack \
--download-slepc \
--download-cmake=externalpackages/cmake-3.24.2.tar.gz \
--download-mumps \
--download-suitesparse=externalpackages/SuiteSparse-5.13.0.tar.gz \
--download-hdf5=externalpackages/hdf5-1.12.1.tar.gz \
--download-make \
--with-packages-download-dir=externalpackages

### Generate compile_commands.json with Bear
I tried Bear on a simple ``HelloWorld'' project, it failed (output a empty json file) when using icc.
A remedy is using PETSC_ARCH=linux-gcc-debug when writing code, and installing Bear with apt-get.
Use the following command to generate compile_commands.json, which is needed by the Clangd plugin of VS Code. 

Bear -- make [a_test_project]

If use CMake, just pass -DCMAKE_EXPORT_COMPILE_COMMANDS=ON into the cmake command.

### Use Valgrind in mpi
Try this command (from https://gist.github.com/v1j4y/d1c3246c7ae764c0165f104d0131e22d):

mpiexec -n 8 valgrind --tool=memcheck -q --num-callers=20 --track-origins=yes --log-file=valgrind.log.%p --dsymutil=yes ./test -malloc off inpfile

Thanks!
