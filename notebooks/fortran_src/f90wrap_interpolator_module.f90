! Module interpolator_module defined in file interpolator_module.f90

subroutine f90wrap_interpolator_initialise(this)
    use interpolator_module, only: interpolator
    implicit none
    
    type interpolator_ptr_type
        type(interpolator), pointer :: p => NULL()
    end type interpolator_ptr_type
    type(interpolator_ptr_type) :: this_ptr
    integer, intent(out), dimension(2) :: this
    allocate(this_ptr%p)
    this = transfer(this_ptr, this)
end subroutine f90wrap_interpolator_initialise

subroutine f90wrap_interpolator_finalise(this)
    use interpolator_module, only: interpolator
    implicit none
    
    type interpolator_ptr_type
        type(interpolator), pointer :: p => NULL()
    end type interpolator_ptr_type
    type(interpolator_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_interpolator_finalise

subroutine f90wrap_init_(this, xc, yc, tc, fx, fy, kx, ky, kt, nx, ny, nt, n0, &
    n1, n2, n3, n4, n5, n6, n7, n8)
    use interpolator_module, only: init_
    implicit none
    
    class(interpolator), intent(inout) :: this
    real(4), intent(in), dimension(n0) :: xc
    real(4), intent(in), dimension(n1) :: yc
    real(4), intent(in), dimension(n2) :: tc
    real(4), intent(in), dimension(n3,n4,n5) :: fx
    real(4), intent(in), dimension(n6,n7,n8) :: fy
    integer, intent(in) :: kx
    integer, intent(in) :: ky
    integer, intent(in) :: kt
    integer, intent(in) :: nx
    integer, intent(in) :: ny
    integer, intent(in) :: nt
    integer :: n0
    !f2py intent(hide), depend(xc) :: n0 = shape(xc,0)
    integer :: n1
    !f2py intent(hide), depend(yc) :: n1 = shape(yc,0)
    integer :: n2
    !f2py intent(hide), depend(tc) :: n2 = shape(tc,0)
    integer :: n3
    !f2py intent(hide), depend(fx) :: n3 = shape(fx,0)
    integer :: n4
    !f2py intent(hide), depend(fx) :: n4 = shape(fx,1)
    integer :: n5
    !f2py intent(hide), depend(fx) :: n5 = shape(fx,2)
    integer :: n6
    !f2py intent(hide), depend(fy) :: n6 = shape(fy,0)
    integer :: n7
    !f2py intent(hide), depend(fy) :: n7 = shape(fy,1)
    integer :: n8
    !f2py intent(hide), depend(fy) :: n8 = shape(fy,2)
    call init_(this=this, xc=xc, yc=yc, tc=tc, fx=fx, fy=fy, kx=kx, ky=ky, kt=kt, &
        nx=nx, ny=ny, nt=nt)
end subroutine f90wrap_init_

subroutine f90wrap_eval_(this, t, x, f, nx, ny, n0, n1, n2, n3)
    use interpolator_module, only: eval_
    implicit none
    
    class(interpolator), intent(inout) :: this
    real(4), intent(in) :: t
    real(4), intent(in), dimension(n0,n1,2) :: x
    real(4), intent(inout), dimension(n2,n3,2) :: f
    integer, intent(in) :: nx
    integer, intent(in) :: ny
    integer :: n0
    !f2py intent(hide), depend(x) :: n0 = shape(x,0)
    integer :: n1
    !f2py intent(hide), depend(x) :: n1 = shape(x,1)
    integer :: n2
    !f2py intent(hide), depend(f) :: n2 = shape(f,0)
    integer :: n3
    !f2py intent(hide), depend(f) :: n3 = shape(f,1)
    call eval_(this=this, t=t, x=x, f=f, nx=nx, ny=ny)
end subroutine f90wrap_eval_

! End of module interpolator_module defined in file interpolator_module.f90

