# 2017 JA
import numpy as np

def lorenz96(tf,x0,Nout):
    """This function computes the time evolution of the Lorenz 96 model.

    It is the general case for N variables; often N=40 is used.

    The Lorenz 1996 model is cyclical: dx[j]/dt=(x[j+1]-x[j-2])*x[j-1]-x[j]+F

    Inputs:  - tf, final time.  Should be a multiple of the time step 0.025.
             - x0, original position.  It can be None and in this case the
               model spins up from rest.
             - N, the number of variables
    Outputs: - t, the time array
             - x, an array of size [len(t) x N]"""

    # Initialize values for integration
    global N, F
    N = Nout
    deltat = 0.025 # the time step 1/40 (to guarantee stability with RK4)
    t = np.arange(0,tf+deltat/2,deltat)
    x = np.empty((len(t),N))
    x.fill(np.nan)

    F = 8 # the forcing, for N>=12 (variables), a forcing F>5 guarantees chaos

    if np.all(x0) == None:
        # If the model is started from rest, then x_j=F for all j and
        # we introduce a perturbation
        x[0,:] = F
        pert = 0.05
        pospert = 1
        x[0,pospert] = F+pert
    else:
        # Started from another initial condition
        x[0,:] = x0

    # The integration
    for i in range(len(t)-1): # for each time
        x[i+1,:] = x[i,:] + rk4(x[i,:],deltat) # solved via RK4

    return t,x

##____________________________________
## Functions for the integration
def rk4(Xold,deltat):
    "The integration method RK4."
    k1 = f(Xold)
    k2 = f(Xold+1/2.0*deltat*k1)
    k3 = f(Xold+1/2.0*deltat*k2)
    k4 = f(Xold+deltat*k3)
    delta = 1/6.0*deltat*(k1+2*k2+2*k3+k4)
    return delta

def f(x):
    "The actual Lorenz 1996 model."
    global N, F
    k=np.empty_like(x); k.fill(np.nan)
    # Remember it is a cyclical model, hence we need modular algebra
    for j in range(N):
        k[j]=(x[(j+1)%N]-x[(j-2)%N])*x[(j-1)%N]-x[j]+F
    return k
    

#############################################################################
def lorenz96_stoch(x_0,tmax,sqrtQ=None):
 #L63 with model error
 # The time array
 tstep = 0.025
 t = np.arange(0,tmax+tstep/2,tstep)
 Nsteps = np.size(t); Nx = np.size(x_0)
 xt = np.empty((Nsteps,Nx)); xt.fill(np.nan)
 xt[0,:] = x_0
 # The cycle containing time integration
 for i in range(Nsteps-1): # for each time step
  Varsold = xt[i,:]
  delta = rk4(Varsold,tstep)
  if sqrtQ==None:
   jump = np.zeros(Nx)
  elif sqrtQ!=None: 
   jump = np.dot(sqrtQ,np.random.randn(Nx))
  x_aux = Varsold + delta + jump
  xt[i+1,:] = x_aux
 del i
 return t,xt




