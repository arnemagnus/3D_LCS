#include <vector>
#include <iostream>

extern "C" void set_handle(void *handle);
extern "C" void release_handle(void *handle);
extern "C" void initialize_interpolator(void *handle, double *x, double *f,
                                                      int kx, int nx,
                                                      int ext);
extern "C" double evaluate_interpolator(void *handle, double x, int dx);

class ItpCont
{
    private:
        void *handle;
    public:
        ItpCont();

        void init_interp(double *x, double *f, int kx, int nx, int ext);

        double eval_interp(double x, int dx);

        ~ItpCont();

};
