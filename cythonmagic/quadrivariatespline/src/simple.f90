module simple
    use bspline_module, only : bspline_4d
    implicit none
    integer, parameter :: wp = kind(1.0d0)

    type :: itp_container
        type(bspline_4d) :: itp
        logical :: initialized
    end type itp_container

contains
    subroutine set_handle(handle) bind(C, name = 'set_handle')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_loc
        type(c_ptr), intent(out) :: handle
        type(itp_container), pointer :: p

        allocate(p)
        handle = c_loc(p)
    end subroutine set_handle

    subroutine destroy_interpolator(handle) bind(C, name='destroy_interpolator')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_f_pointer
        type(c_ptr), intent(in), value :: handle
        type(itp_container), pointer :: p

        call c_f_pointer(handle, p)
        if (p%initialized) then
            call p%itp%destroy()
            p%initialized = .false.
        end if
    end subroutine destroy_interpolator

    subroutine initialize_interpolator(handle, x, y, z, q, f, kx, ky, kz, kq,&
                                             & nx,ny,nz,nq, ext) &
                                    & bind(C, name = 'initialize_interpolator')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_f_pointer, c_int, c_double, c_bool
        type(c_ptr), intent(in), value :: handle
        integer(c_int), intent(in), value :: nx, ny, nz,nq, kx, ky, kz, kq
        real(c_double), intent(in) :: x(nx), y(ny), z(nz), q(nq), f(nx,ny,nz,nq)
        integer(c_int), intent(in), value :: ext
        type(itp_container), pointer :: p
        logical :: extrap
        integer :: iflag

        if (ext == 1) then
            extrap = .true.
        else
            extrap = .false.
        end if

        call c_f_pointer(handle, p)
        call p%itp%initialize(x,y,z,q,f,kx,ky,kz,kq,iflag,extrap)
        p%initialized = .true.
    end subroutine initialize_interpolator

    function evaluate_interpolator(handle, x, y, z, q, dx, dy, dz, dq) result (f) &
                                    & bind(C, name = 'evaluate_interpolator')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_f_pointer, c_int, c_double
        type(c_ptr), intent(in), value :: handle
        real(c_double), intent(in), value :: x, y, z, q
        integer(c_int), intent(in), value :: dx, dy, dz, dq
        real(c_double) :: f
        type(itp_container), pointer :: p
        integer :: iflag

        call c_f_pointer(handle, p)
        call p%itp%evaluate(x,y,z,q,dx,dy,dz,dq,f,iflag)
    end function evaluate_interpolator

    subroutine release_handle(handle) bind(C, name='release_handle')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_f_pointer
        type(c_ptr), intent(in), value :: handle
        type(itp_container), pointer :: p

        call c_f_pointer(handle, p)
        if (p%initialized) then
            call p%itp%destroy()
        end if
        deallocate(p)
    end subroutine release_handle


end module simple
