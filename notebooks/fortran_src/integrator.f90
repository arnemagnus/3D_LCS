module integrator_module
    use interpolator_module, only : interpolator
    use parameters, only : wp
    implicit none
contains
    subroutine rk2(t,x,h,f,nx,ny)
        implicit none
        integer, intent(in) :: nx,ny
        real(wp), intent(inout) :: t,x(nx,ny,2)
        real(wp), intent(in)    :: h
        type(interpolator), intent(inout) :: f
        real(wp)                :: k1(nx,ny,2), k2(nx,ny,2)
        call f%eval(t,x,k1,nx,ny)
        call f%eval(t+h,x+k1*h,k2,nx,ny)
        x = x + (k1+k2)*h/2._wp
        t = t + h
    end subroutine rk2
end module integrator_module
