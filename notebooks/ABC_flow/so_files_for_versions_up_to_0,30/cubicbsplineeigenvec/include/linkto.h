#include <vector>
#include <iostream>

extern "C" void set_handle(void *handle);
extern "C" void release_handle(void *handle);
extern "C" void initialize_interpolator(void *handle, double *x, double *y,
                                                      double *z, double *f,
                                                      int kx, int ky, int kz,
                                                      int nx, int ny, int nz,
                                                      int ext);
extern "C" void destroy_interpolator(void *handle);
extern "C" double evaluate_interpolator(void *handle,
                                        double x, double y, double z,
                                        int dx, int dy, int dz);

class ItpCont
{
    private:
        void *handle;
    public:
        ItpCont();

        void init_interp(double *x, double *y, double *z, double *f,
                         int kx, int ky, int kz, int nx, int ny, int nz, int ext);

        double eval_interp(double x, double y, double z, int dx, int dy, int dz);

        void kill_interp();

        ~ItpCont();

};
