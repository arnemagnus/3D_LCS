#include "linkto.h"

ItpCont::ItpCont(){
    set_handle(&handle);
};

ItpCont::~ItpCont(){release_handle(handle);};



void ItpCont::init_interp(double *x, double *y, double *z, double *q, double *f,
                          int kx, int ky, int kz, int kq, int nx, int ny, int nz, int nq, int ext){
    ::initialize_interpolator(handle, x, y, z, q, f, kx, ky, kz, kq, nx, ny, nz, nq, ext);
}

double ItpCont::eval_interp(double x, double y, double z, double q, int dx, int dy, int dz, int dq){
    return ::evaluate_interpolator(handle, x, y, z, q, dx, dy, dz, dq);
}

void ItpCont::kill_interp(){
    ::destroy_interpolator(handle);
}

