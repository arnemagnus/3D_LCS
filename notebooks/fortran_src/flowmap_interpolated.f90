module flowmap_interpolated
   use parameters, only : wp
   use interpolator_module, only : interpolator
   use integrator_module, only: rk2
   use doublegyreflow, only : vel
   implicit none
contains
    subroutine flowmap_endpoints(amp,eps,omg,x,x_end,x_samp,ts,tf,h,t_samp,kx,ky,kt,nx,ny,nx_s,ny_s,nt_s)
        implicit none
        integer, intent(in) :: kx,ky,kt,nx,ny,nx_s,ny_s,nt_s
        real(wp), intent(in) :: x(nx,ny,2)
        real(wp), intent(in) :: x_samp(nx_s,ny_s,2),t_samp(nt_s),&
                                &amp,eps,omg,ts,tf,h
        real(wp), intent(out) :: x_end(nx,ny,2)
        real(wp)            :: gfx(nt_s,nx_s,ny_s), gfy(nt_s,nx_s,ny_s), &
                               & v(nx_s,ny_s,2),t
        type(interpolator)  :: f
        integer :: i
        ! Sample velocity field
        do i = 1, nt_s
            call vel(t_samp(i),x_samp,amp,eps,omg,nx_s,ny_s,v)
            gfx(i,:,:) = v(:,:,1)
            gfy(i,:,:) = v(:,:,2)
        end do

        ! Create interpolator from discrete data
        call f%init(x_samp(:,1,1),x_samp(1,:,2),t_samp,gfx,gfy,kt,kx,ky,nx_s,ny_s,nt_s)

        x_end = x
        ! Advect tracer particles
        t = ts
        do while(t<tf)
            call rk2(t,x_end,h,f,nx,ny)
        end do
    end subroutine flowmap_endpoints
end module flowmap_interpolated
