#include <vector>
#include <iostream>

extern "C" void set_handle(void *handle);
extern "C" void release_handle(void *handle);
extern "C" void initialize_interpolator(void *handle, double *x, double *y,
                                                      double *z, double *q,
                                                      double *f,
                                                      int kx, int ky, int kz, int kq,
                                                      int nx, int ny, int nz, int nq,
                                                      int ext);
extern "C" void destroy_interpolator(void *handle);
extern "C" double evaluate_interpolator(void *handle,
                                        double x, double y, double z, double q,
                                        int dx, int dy, int dz, int dq);

class ItpCont
{
    private:
        void *handle;
    public:
        ItpCont();

        void init_interp(double *x, double *y, double *z, double *q, double *f,
                         int kx, int ky, int kz, int kq, int nx, int ny, int nz, int nq, int ext);

        double eval_interp(double x, double y, double z, double q, int dx, int dy, int dz, int dq);

        void kill_interp();

        ~ItpCont();

};
