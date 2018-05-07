ANACONDA_ROOT_DIR = /home/arne/anaconda3
FORTRAN_INC_DIR = /usr/local/include
FORTRAN_LIB_DIR = /usr/local/lib
FORTRAN_LIB = bspline-fortran
PYTHON_VERSION = 3.6
LOCAL_INC_DIR = ./include
NUMPY_INC_DIR = ${ANACONDA_ROOT_DIR}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include

FORTRAN_COMP = gfortran
C_COMP = gcc
CXX_COMP = g++

LOCAL_SRC_DIR = ./src

BUILD_DIR = ./build/tmp


all: cython fortran cxx

cython:
	mkdir -p ${BUILD_DIR}
	cython ${LOCAL_SRC_DIR}/straininterpolationandintegration.pyx --cplus -o ${BUILD_DIR}/straininterpolationandintegration.cpp

fortran:
	mkdir -p ${BUILD_DIR}
	${FORTRAN_COMP} -c ${LOCAL_SRC_DIR}/simple.f90 -o ${BUILD_DIR}/simple.o -O3\
		-fPIC -I${FORTRAN_INC_DIR} -L${FORTRAN_LIB_DIR} -l${FORTRAN_LIB}

cxx:
	mkdir -p  ${BUILD_DIR}
	${C_COMP} -pthread -B ${ANACONDA_ROOT_DIR}/compiler_compat -Wl,--sysroot=/ \
		-Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I${NUMPY_INC_DIR} \
		-I${FORTRAN_INC_DIR} -I${LOCAL_INC_DIR} \
		-I${ANACONDA_ROOT_DIR}/include/python${PYTHON_VERSION}m \
		-I${NUMPY_INC_DIR} \
		-c ${BUILD_DIR}/straininterpolationandintegration.cpp -o ${BUILD_DIR}/straininterpolationandintegration.o
	${C_COMP} -pthread -B ${ANACONDA_ROOT_DIR}/compiler_compat -Wl,--sysroot=/ \
		-Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I${NUMPY_INC_DIR} \
		-I${FORTRAN_INC_DIR} -I${LOCAL_INC_DIR} \
		-I${ANACONDA_ROOT_DIR}/include/python${PYTHON_VERSION}m \
		-I${NUMPY_INC_DIR} \
		-c ${LOCAL_SRC_DIR}/linkto.cpp -o ${BUILD_DIR}/linkto.o
	${CXX_COMP} -pthread -shared -B ${ANACONDA_ROOT_DIR}/compiler_compat \
		-L${ANACONDA_ROOT_DIR}/lib -Wl,-rpath=/${ANACONDA_ROOT_DIR}/lib \
		-O3 -Wl,--no-as-needed -Wl,--sysroot=/ ${BUILD_DIR}/straininterpolationandintegration.o \
		${BUILD_DIR}/linkto.o ${BUILD_DIR}/simple.o -o ./straininterpolationandintegration.so \
		-L${FORTRAN_LIB_DIR} -I${FORTRAN_INC_DIR} -I${NUMPY_INC_DIR} \
		-l${FORTRAN_COMP} -l${FORTRAN_LIB}

clean:
	rm -rf ./build
	rm *.so
