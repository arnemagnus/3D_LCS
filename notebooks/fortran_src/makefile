OUTPUT_PYHDR = pymod.pyx
OUTPUT_PYMOD = pymod_velfield
OUTPUT_FORTRAN = velfield_debug.out
MAIN_PY_SRC = flowmap_interpolated.f90
MAIN_FO_SRC = debug_main.f90
AUX_PY_SRC = parameters.f90 doublegyreflow.f90 interpolator_module.f90 integrator.f90
AUX_FO_SRC = $(AUX_PY_SRC) $(MAIN_PY_SRC)
COMPILED_SRCS = *.o
OPTIMIZATION = -O3
LIB_PATH = /usr/local/lib/
INC_PATH = /usr/local/include/
LIBS = -lopenblas -llapack95 -lbspline-fortran
COMP_FLAGS = -fPIC -fopenmp -lgomp -shared
fortran:
	gfortran -shared -c $(COMP_FLAGS) $(OPTIMIZATION) $(AUX_FO_SRC) -L $(LIB_PATH) -I $(INC_PATH) $(LIBS)
	gfortran -shared -o $(OUTPUT_FORTRAN) $(MAIN_FO_SRC) $(COMPILED_SRCS) $(OPTIMIZATION) $(COMP_FLAGS) -L $(LIB_PATH) -I $(INC_PATH) $(LIBS)
pymod:
	gfortran -shared -c $(COMP_FLAGS) $(OPTIMIZATION) $(AUX_PY_SRC) -L $(LIB_PATH) -I $(INC_PATH) $(LIBS)
	f2py --verbose -m $(OUTPUT_PYMOD) --fcompiler=gfortran --f90flags='$(COMP_FLAGS)' --opt='$(OPTIMIZATION)' -c $(MAIN_PY_SRC) $(COMPILED_SRCS) -L/$(LIB_PATH) -I/$(INC_PATH) $(LIBS)
hdr_pymod:
	gfortran -shared -c $(COMP_FLAGS) $(OPTIMIZATION) $(AUX_PY_SRC) -L $(LIB_PATH) -I $(INC_PATH) $(LIBS)
	f2py --verbose -h $(OUTPUT_PYHDR) --fcompiler=gfortran --f90flags='$(COMP_FLAGS)' --opt='$(OPTIMIZATION)' -c $(MAIN_PY_SRC) $(COMPILED_SRCS) -L/$(LIB_PATH) -I/$(INC_PATH) $(LIBS)
clean:
	rm *.so -f
	rm *.mod -f
	rm *.o -f
	rm *.out -f
