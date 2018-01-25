import _derp
import f90wrap.runtime
import logging

class Interpolator_Module(f90wrap.runtime.FortranModule):
    """
    Module interpolator_module
    
    
    Defined at interpolator_module.f90 lines 1-36
    
    """
    class Interpolator(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=interpolator)
        
        
        Defined at interpolator_module.f90 lines 5-10
        
        """
        def __init__(self, handle=None):
            """
            self = Interpolator()
            
            
            Defined at interpolator_module.f90 lines 5-10
            
            
            Returns
            -------
            this : Interpolator
            	Object to be constructed
            
            
            Automatically generated constructor for interpolator
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            self._handle = _derp.f90wrap_interpolator_initialise()
        
        def __del__(self):
            """
            Destructor for class Interpolator
            
            
            Defined at interpolator_module.f90 lines 5-10
            
            Parameters
            ----------
            this : Interpolator
            	Object to be destructed
            
            
            Automatically generated destructor for interpolator
            """
            if self._alloc:
                _derp.f90wrap_interpolator_finalise(this=self._handle)
        
        _dt_array_initialisers = []
        
    @staticmethod
    def init_(this, xc, yc, tc, fx, fy, kx, ky, kt, nx, ny, nt):
        """
        init_(this, xc, yc, tc, fx, fy, kx, ky, kt, nx, ny, nt)
        
        
        Defined at interpolator_module.f90 lines 13-20
        
        Parameters
        ----------
        this : unknown
        xc : float array
        yc : float array
        tc : float array
        fx : float array
        fy : float array
        kx : int
        ky : int
        kt : int
        nx : int
        ny : int
        nt : int
        
        """
        _derp.f90wrap_init_(this=this, xc=xc, yc=yc, tc=tc, fx=fx, fy=fy, kx=kx, ky=ky, \
            kt=kt, nx=nx, ny=ny, nt=nt)
    
    @staticmethod
    def eval_(this, t, x, f, nx, ny):
        """
        eval_(this, t, x, f, nx, ny)
        
        
        Defined at interpolator_module.f90 lines 22-36
        
        Parameters
        ----------
        this : unknown
        t : float
        x : float array
        f : float array
        nx : int
        ny : int
        
        """
        _derp.f90wrap_eval_(this=this, t=t, x=x, f=f, nx=nx, ny=ny)
    
    _dt_array_initialisers = []
    

interpolator_module = Interpolator_Module()

