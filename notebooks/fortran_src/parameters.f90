module parameters
    implicit none
    integer, parameter  :: sp = kind(1.0E0) ! Single precision
    integer, parameter  :: dp = kind(1.0D0) ! Double precision
    integer, parameter  :: wp = dp          ! Working precision

    real(wp), parameter :: pi = 4._wp*atan(1._wp)
end module parameters
