#include "linkto.h"

ItpCont::ItpCont(){
//    handle = NULL;
    set_handle(&handle);
};

ItpCont::~ItpCont(){release_handle(handle);};



void ItpCont::init_interp(double *x, double *f, int kx, int nx, int ext){
    ::initialize_interpolator(handle, x, f, kx, nx, ext);
}

double ItpCont::eval_interp(double x, int dx){
    return ::evaluate_interpolator(handle, x, dx);
}
