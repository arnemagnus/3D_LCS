module OTmod
    use f95_lapack
    implicit none
contains
    subroutine kek()
        implicit none
        integer, parameter :: wp = kind(1.0D0)
        real(wp) :: A(2,2), b(2)
        integer :: i
        A = 0._wp
        do i = 1, 2
            A(i,i) = 1._wp
        end do

        b = (/0.5_wp, 0.25_wp/)

        call la_gesv(A,b)

        write(*,*) 'fagt'
    end subroutine kek
end module OTmod
