#include "linkto.h"

ItpCont::ItpCont(){
    set_handle(&handle);
};

ItpCont::~ItpCont(){release_handle(handle);};



void ItpCont::init_interp(double *x, double *y, double *z, double *f,
                          int kx, int ky, int kz, int nx, int ny, int nz, int ext){
    ::initialize_interpolator(handle, x, y, z, f, kx, ky, kz, nx, ny, nz, ext);
}

double ItpCont::eval_interp(double x, double y, double z, int dx, int dy, int dz){
    return ::evaluate_interpolator(handle, x, y, z, dx, dy, dz);
}

void ItpCont::kill_interp(){
    ::destroy_interpolator(handle);
}
