# PETSc information.
PETSC_DIR = $(HOME)/petsc
PETSC_ARCH = linux-gcc-debug
include ${PETSC_DIR}/lib/petsc/conf/variables
# include ${PETSC_DIR}/lib/petsc/conf/rules

# Thanks to Job Vranish (https://spin.atomicobject.com/2016/08/26/makefile-c-projects/).
TARGET_EXEC := test

ROOT_DIR := $(shell pwd)
BUILD_DIR := $(ROOT_DIR)/build
SRC_DIR := $(ROOT_DIR)/src

# Find all the C and C++ files we want to compile.
# Note the single quotes around the * expressions. Make will incorrectly expand these otherwise.
# SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')
SRC_FILES := test.c PreMixFEM_3D.c

# String substitution for every C/C++ file.
# As an example, hello.cpp turns into ./build/hello.cpp.o


# String substitution (suffix version without %).
# As an example, ./build/hello.cpp.o turns into ./build/hello.cpp.d
# DEPS := $(OBJS:.o=.d)

# Every folder in ./src will need to be passed to GCC so that it can find header files
# INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
# INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.

# Build step for C source

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(SRC_FILES:%.c=$(BUILD_DIR)/%.o)
#	@-/usr/bin/printf '\n' 
#	@-echo $(C_SH_LIB_PATH)
#	@-/usr/bin/printf '\n'  
#	@-echo $(PETSC_LIB_DIR)
#	@-/usr/bin/printf '\n' 
#	@-echo $(PETSC_TS_LIB_BASIC)
#	@-/usr/bin/printf '\n' 
#	@-echo $(PETSC_EXTERNAL_LIB_BASIC)
#	@-/usr/bin/printf '\n' 
#	@-echo $(LINK.C)
	$(CXX_LINKER) $(CXX_LINKER_FLAGS) -o $@ $^ $(PETSC_TS_LIB) -lslepc

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
#	mkdir -p $(dir $@)
	$(CC) ${CC_FLAGS} -I$(SRC_DIR) ${PETSC_CC_INCLUDES} -c $< -o $@


# Build step for C++ source
# $(BUILD_DIR)/%.cpp.o: %.cpp
# 	mkdir -p $(dir $@)
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# .PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*.o

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
# -include $(DEPS)



#  To access the PETSc variables for the build, including compilers, compiler flags, libraries etc but
#  manage the build rules yourself (rarely needed) comment out the next lines
# include ${PETSC_DIR}/lib/petsc/conf/rules
# include ${PETSC_DIR}/lib/petsc/conf/test




