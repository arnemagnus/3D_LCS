module doublegyreflow
    use parameters, only: wp, pi
    implicit none
contains
    subroutine a_(t,eps,omg,a)
        implicit none
        real(wp), intent(in)    :: t,eps,omg
        real(wp), intent(out)   :: a
        a = eps*sin(omg*t)
    end subroutine a_
    subroutine b_(t,eps,omg,b)
        implicit none
        real(wp), intent(in)    :: t,eps,omg
        real(wp), intent(out)   :: b
        b = 1 - 2*eps*sin(omg*t)
    end subroutine b_
    subroutine f_(t,x,amp,eps,omg,nx,ny,f)
        implicit none
        integer, intent(in)     :: nx,ny
        real(wp), intent(in)    :: t,x(nx,ny,2),amp,eps,omg
        real(wp), intent(out)   :: f(nx,ny)
        real(wp)                :: a,b
        integer                 :: i
        call a_(t,eps,omg,a)
        call b_(t,eps,omg,b)
        do i = 1, nx
        f(i,:) = a*x(i,:,1)**2 + b*x(i,:,1)
        end do
    end subroutine f_
    subroutine dfdx_(t,x,amp,eps,omg,nx,ny,df)
        implicit none
        integer, intent(in)     :: nx,ny
        real(wp), intent(in)    :: t,x(nx,ny,2),amp,eps,omg
        real(wp), intent(out)   :: df(nx,ny)
        real(wp)                :: a,b
        integer                 :: i
        call a_(t,eps,omg,a)
        call b_(t,eps,omg,b)
        do i = 1, nx
        df(i,:) = 2*a*x(i,:,1) + b
        end do
    end subroutine dfdx_
    subroutine vel(t,x,amp,eps,omg,nx,ny,v)
        implicit none
        integer, intent(in)     :: nx,ny ! Grid dimensions
        real(wp), intent(in)    :: t,x(nx,ny,2),amp,eps,omg
        real(wp), intent(out)   :: v(nx,ny,2)
        real(wp)                :: f(nx,ny),df(nx,ny)
        call f_(t,x,amp,eps,omg,nx,ny,f)
        call dfdx_(t,x,amp,eps,omg,nx,ny,df)
        v(:,:,1) = -pi*amp*sin(pi*f)*cos(pi*x(:,:,2))
        v(:,:,2) = pi*amp*cos(pi*f)*sin(pi*x(:,:,2))*df
    end subroutine vel
end module doublegyreflow
