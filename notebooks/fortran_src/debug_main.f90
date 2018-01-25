program main
    use parameters, only : wp,pi
    use interpolator_module, only : interpolator
    use flowmap_interpolated, only : flowmap_endpoints
    implicit none

    integer, parameter :: nx_s = 500
    integer, parameter :: ny_s = 250
    integer, parameter :: nt_s = 300

    integer, parameter :: nx = 200
    integer, parameter :: ny = 100

    real(wp) :: dx_s,dy_s,dt_s
    real(wp) :: t, x(nx,ny,2), dx, dy, dt, x_samp(nx_s,ny_s,2),tc(nt_s)

    type(interpolator) :: f

    real(wp), parameter :: amp = 0.1_wp, eps = 0.1_wp, omg = 2._wp*pi/10._wp,&
                            &ts = 0._wp,tf = 20._wp,h=0.01_wp
    integer :: i, j, k
    dx_s = 2.0_wp/(nx_s-1)
    dy_s = 1.0_wp/(ny_s-1)
    dt_s = 30._wp/(nt_s-1)

    dx = 2.0_wp/(nx-1)
    dy = 1.0_wp/(ny-1)

    do i = 1, nx_s
        do j = 1, ny_s
        x_samp(i,j,1) = (i-1)*dx_s
        x_samp(i,j,2) = (j-1)*dy_s
        end do
    end do
    do k = 1, nt_s
        tc(k) = (k-1)*dt_s
    end do

    do i = 1, nx
    do j = 1, ny
        x(i,j,1) = (i-1)*dx
        x(i,j,2) = (j-1)*dy
    end do
    end do

    write(*,*) 'Pre-advection'
    call flowmap_endpoints(amp,eps,omg,x,x,x_samp,ts,tf,h,tc,3,3,3,nx,ny,nx_s,ny_s,nt_s)
    write(*,*) 'Post-advection'
end program main
