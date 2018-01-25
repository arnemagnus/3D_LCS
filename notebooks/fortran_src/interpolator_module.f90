module interpolator_module
    use bspline_module, only: bspline_3d
    use parameters, only: wp
    implicit none
    type :: interpolator
        type(bspline_3d) :: itpx,itpy
    contains
        private
        procedure, public :: eval => eval_
        procedure, public :: init => init_
    end type interpolator
contains
    subroutine init_(this,xc,yc,tc,fx,fy,kx,ky,kt,nx,ny,nt)
        implicit none
        class(interpolator), intent(inout)  :: this
        integer, intent(in) :: kx,ky,kt,nx,ny,nt
        real(wp), intent(in) :: xc(nx),yc(nx),tc(nt),fx(nt,nx,ny),fy(nt,nx,ny)
        integer     :: iflag
        call this%itpx%initialize(tc,xc,yc,fx,kt,kx,ky,iflag)
        call this%itpy%initialize(tc,xc,yc,fy,kt,kx,ky,iflag)
    end subroutine init_
    subroutine eval_(this,t,x,f,nx,ny)
        implicit none
        class(interpolator), intent(inout) :: this
        integer, intent(in) :: nx,ny
        real(wp), intent(in) :: t,x(nx,ny,2)
        real(wp), intent(out) :: f(nx,ny,2)
        integer :: iflag, i, j
        integer, parameter  :: dt = 0, dx = 0, dy = 0

        do i = 1, nx
        do j = 1, ny
        call this%itpx%evaluate(t,x(i,j,1),x(i,j,2),dt,dx,dy,f(i,j,1),iflag)
        call this%itpx%evaluate(t,x(i,j,1),x(i,j,2),dt,dx,dy,f(i,j,2),iflag)
        end do
        end do
    end subroutine eval_
end module interpolator_module
