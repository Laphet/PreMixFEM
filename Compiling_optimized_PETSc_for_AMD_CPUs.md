### Introduction
In this experiment, we want to compile an optimized PETSc library for AMD 7452 CPUs. We follow the AMD document [HPC tuning guide for 7002 series processors] to build optimized libraries, and we expected it may finally accelerate the computing time of our numerical experiments.


### Set compiling flags
#!/bin/sh  

unset PETSC_DIR  
unset PETSC_ARCH  
module purge  
export CC=clang  
export FC=flang  
export CXX=clang++  
export F90=flang  
export F77=flang  
export AR=llvm-ar  
export RANLIB=llvm-ranlib  
export NM=llvm-nm  
export COMPILERROOT=${HOME}/cqye/aocc-compiler-4.0.0  
export OMPI=${COMPILERROOT}/include  
export OMPL=${COMPILERROOT}/lib  

export CFLAGS="-Ofast -ffast-math -march=znver2 -fopenmp -I${OMPI}"  
export CXXFLAGS="-Ofast -ffast-math -march=znver2 -fopenmp -I${OMPI}"  
export FCFLAGS="-Ofast -ffast-math -march=znver2 -fopenmp -I${OMPI}"  
export LDFLAGS="-L${OMPL}"  
export INCLUDE=${OMPI}:${INCLUDE}  
export PATH=${COMPILERROOT}/bin:${PATH}  
export LD_LIBRARY_PATH=${OMPL}:${LD_LIBRARY_PATH}  

export JEMALLOCROOT=${HOME}/cqye/jemalloc-dev/jemalloc_amd_opt  
export KNEMROOT=${HOME}/cqye/knem-1.1.4/knem_amd_opt  
export OPENMPIROOT=${HOME}/cqye/openmpi-4.1.4/openmpi_amd_opt
export PATH=${OPENMPIROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${OPENMPIROOT}/lib:${LD_LIBRARY_PATH}

export BLIS_ROOT_PATH=${HOME}/cqye/amd-blis
export LIBFLAME_ROOT_PATH=${HOME}/cqye/amd-libflame


### Build Jemalloc
export JEMALLOCROOT=${HOME}/cqye/jemalloc-dev/jemalloc_amd_opt  
./autogen.sh  
CFLAGS=${CFLAGS} ./configure --prefix=${JEMALLOCROOT}  
make -j  
make install  


### Build KNEM
export KNEMROOT=${HOME}/cqye/knem-1.1.4/knem_amd_opt  
./autogen.sh  
./configure --prefix=${KNEMROOT} CFLAGS="${CFLAGS} -I${JEMALLOCROOT}/include/jemalloc" \
LDFLAGS="${LDFLAGS} -L${JEMALLOCROOT}/lib -ljemalloc" --host=x86_64  
make clean  
make  
make install  


### Build OpenMPI
Note the cluster is managed by Slurm, and configure OpenMPI with "--with-slurm".  

export OPENMPIROOT=${HOME}/cqye/openmpi-4.1.4/openmpi_amd_opt  
./configure --prefix=${OPENMPIROOT}  --with-knem=${KNEMROOT} --with-slurm \
CC=${CC} CXX=${CXX} FC=${FC} CFLAGS="${CFLAGS}" \
CXXFLAGS="${CXXFLAGS}" FCFLAGS="${FCFLAGS}" \
--enable-mpi-fortran --enable-shared=yes --enable-static=yes --enable-mpi1-compatibility  
make -j  
make install  
export PATH=${OPENMPIROOT}/bin:${PATH}  
export LD_LIBRARY_PATH=${OPENMPIROOT}/lib:${LD_LIBRARY_PATH}  

### Build PETSc
./configure PETSC_ARCH=linux-amd-opt --with-debugging=0 \  
--with-mpi-dir=${OPENMPIROOT} \  
--CFLAGS='-Ofast -fopenmp -march=znver2 -ffast-math -funroll-loops' \  
--FFLAGS='-Ofast -fopenmp -march=znver2 -ffast-math -funroll-loops' \  
--CXXFLAGS='-Ofast -fopenmp -march=znver2 -ffast-math -funroll-loops' \  

--with-blaslapack-dir=${MKLROOT} \  
--download-scalapack=externalpackages/scalapack-2.2.0.tar.gz \  
--download-slepc=externalpackages/slepc-3811.tar.gz \  
--download-mumps=externalpackages/MUMPS_5.5.1.tar.gz \  
--download-suitesparse=externalpackages/SuiteSparse-5.13.0.tar.gz \  
--download-hdf5=externalpackages/hdf5-1.12.1.tar.gz \  
--download-sowing=externalpackages/petsc-pkg-sowing-93e363f25328.tar.gz \  
--with-mkl_cpardiso-dir=${MKLROOT} \  
--download-metis=externalpackages/petsc-pkg-metis.tar.gz \  
--download-parmetis=externalpackages/petsc-pkg-parmetis.tar.gz \  
--download-superlu_dist=externalpackages/superlu_dist-8.1.2.tar.gz















