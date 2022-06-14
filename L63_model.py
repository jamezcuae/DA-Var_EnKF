# 2017 JA
import numpy as np

def lorenz63(x_0,tmax,paramout=None):
    """Evolution of the Lorenz 1963 3-variable model.

    Inputs:  - x_0, an array containing the initial position
             - tmax, the maximum time.  Should be a multiple of the
               timestep 0.01.
    Outputs: - t, the time array
             - xt, the nature run [len(t) x 3]"""
    # The time array
    tstep = 0.01
    t = np.arange(0,tmax+tstep/2,tstep)
    # Note that with this step size, 8 steps are approximately linear windows
    # Now, define the vectors for space vars. They're organized in an
    # array of 3 columns [x,y,z].
    xt = np.empty((len(t),3))
    xt.fill(np.nan)
    xt[0,:] = x_0

    # The cycle containing time integration
    for i in range(len(t)-1): # for each time step
        Varsold = xt[i,:]
        # We integrate using Runge-Kutta-4
        Varsnew = rk4(Varsold,tstep,paramout)
        xt[i+1,:] = Varsnew

    return t,xt

## Auxiliary functions
def rk4(Varsold,tstep,paramout=None):
    "This function contains the RK4 routine."
    k1 = f(Varsold,paramout)
    k2 = f(Varsold+1/2.0*tstep*k1,paramout)
    k3 = f(Varsold+1/2.0*tstep*k2,paramout)
    k4 = f(Varsold+tstep*k3,paramout)
    Varsnew = Varsold + 1/6.0*tstep*(k1+2*k2+2*k3+k4)
    return Varsnew

def f(x,paramout=None):
    "This function contains the actual L63 model."
    # The parameters
    if np.all(paramout)==None:
     sigma = 10.0;     b = 8/3.0;    r = 28.0
    else: 
     sigma = paramout[0];  b = paramout[1];  r = paramout[2]
     
    # Initialize
    k = np.empty_like(x)
    k.fill(np.nan)
    # The Lorenz equations
    k[0] = sigma*(x[1]-x[0])
    k[1] = x[0]*(r-x[2])-x[1]
    k[2] = x[0]*x[1]-b*x[2]
    return k


#############################
def lorenz63_stoch(x_0,tmax,sqrtQ=None):
 #L63 with model error
 # The time array
 tstep = 0.01
 t = np.arange(0,tmax+tstep/2,tstep)
 Nsteps = np.size(t)
 Nx = np.size(x_0)
 xt = np.empty((Nsteps,Nx)); xt.fill(np.nan)
 xt[0,:] = x_0
 # The cycle containing time integration
 for i in range(Nsteps-1): # for each time step
  Varsold = xt[i,:]
  Varsnew = rk4(Varsold,tstep)
  if np.all(sqrtQ)==None:
   jump = [0,0,0]
  else: 
   jump = np.dot(sqrtQ,np.random.randn(Nx))
  x_aux = Varsnew + jump
  xt[i+1,:] = x_aux
 del i
 return t,xt



