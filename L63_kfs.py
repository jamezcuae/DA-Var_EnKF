# 2017 JA
import numpy as np
from scipy.linalg import pinv, sqrtm
from L63_model import lorenz63

def kfs_lor63(x0_guess,t,tobs,y,H,R,rho,M,met):
 """Data assimilation for Lorenz 1963 using Ensemble Kalman Filters.
 Inputs:  - x0_t, the real initial position
          - t, time array of the model (should be evenly spaced)
          - tobs, time array of the observations (should be evenly
            spaced with a timestep that is a multiple of the model timestep)
          - y, the observations
          - H, observation matrix
          - R, the observational error covariance matrix
          - rho, inflation for P. We multiply (1+rho)*Xpert or P*(1+rho)^2.
          - M, the ensemble size
          - met, a string containing the method: 'SEnKF', 'ETKF'
 Outputs: - Xb, the background ensemble 3D array [time,vars,members]
          - xb, background mean
          - Xa, the analysis ensemble 3D array [time,vars,members]
          - xa, analysis mean"""

 # General settings
 # Number of variables
 N = np.size(x0_guess)
 # For the true time
 Nsteps = np.size(t)
 tstep_truth = t[1]-t[0]
 # For the analysis (we assimilate everytime we get observations)
 tstep_obs = tobs[1]-tobs[0]
 # The ratio
 o2t = int(tstep_obs/tstep_truth+0.5)
 # Precreate the arrays for background and analysis
 Xb = np.empty((len(t),N,M)); Xb.fill(np.nan)
 Xa = np.empty((len(t),N,M))
 Xa.fill(np.nan)

 # For the original background ensemble
 back0 = 'fixed' # so we can generate repeatable experiments.
 desv = 2.0
 # Fixed initial conditions for our ensemble (created ad hoc)
 if back0=='fixed':
  for j in range(N):
   Xb[0,j,:] = np.linspace(x0_guess[j]-desv,x0_guess[j]+desv,M)
  del j
 # Random initial conditions for our ensemble; let's perturb the
 # real x0 using errors with the magnitudes of R
 elif back0=='random':
  for j in range(M):
   Xb[0,:,j] = x0_guess + desv*np.random.randn(N)
  del j
  
 # Since we don't have obs at t=0 the first analysis is the same as
 # background
 Xa[0,:,:] = Xb[0,:,:]

 # The following cycle contains evolution and assimilation for all
 # the time steps
 for j in range(np.size(tobs)-1):
  # First we evolve the ensemble members
  # Evolve from analysis!
  xold = Xa[j*o2t,:,:] # [N,M]
  # Time goes forward
  xnew = evolvemembers(xold,tstep_truth,o2t) # needs [N,M] arrays,
                                                   # gives [o2t+1,N,M]
  # The new background
  Xb[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  Xa[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  # The assimilation
  Xa_aux = enkfs(Xb[(j+1)*o2t,:,:],y[j+1,:],H,R,rho,met)
                                                   # gets a [N,M] array
  Xa[(j+1)*o2t,:,:] = Xa_aux # introduce the auxiliary variable
                                 # in the array
  #print('t=', t[(j+1)*o2t])
 del j
 
 # The background and analysis mean
 x_b = np.mean(Xb,axis=2) # [t,N,M] -> [t,N]
 x_a = np.mean(Xa,axis=2) # [t,N,M] -> [t,N]

 return Xb,x_b,Xa,x_a



##################################################################
def evolvemembers(xold,tstep_truth,o2t,parout=None):
 """Evolving the members.
 Inputs:  - xold, a [N,M] array of initial conditions for the
            M members and N variables
          - tstep_truth, the time step used in the nature run
          - o2t, frequency of observations in time steps
 Outputs: - xnew, a [o2t+1,N,M] array with the evolved members"""
 t_anal = o2t*tstep_truth
 N,M = np.shape(xold)
 xnew = np.empty((o2t+1,N,M))
 xnew.fill(np.nan)
 for j in range(M):
  if np.all(parout)==None:
   taux,xaux = lorenz63(xold[:,j],t_anal) # [o2t+1,N]
  else:      
   taux,xaux = lorenz63(xold[:,j],t_anal,parout[:,j]) # [o2t+1,N]
  xnew[:,:,j] = xaux
 del j 
 return xnew

## The EnKF algorithms
def enkfs(Xb,y,H,R,rho,met):
    """Performs the analysis using different EnKF methods.

    Inputs: - Xb, the ensemble background [N,M]
            - y, the observations [L]
            - H, the observation matrix [L,N]
            - R, the obs error covariance matrix [L,L]
            - rho, inflation for P.  Notice we multiply (1+rho)*Xpert
              or P*(1+rho)^2.
            - met, a string that indicated what method to use
    Output: - Xa, the full analysis ensemble [N,M]"""

    # General settings
    # The background information
    Xb = np.mat(Xb) # array -> matrix
    y = np.mat(y).T # array -> column vector

    # Number of state variables, ensemble members and observations
    N,M = Xb.shape
    L,N = H.shape

    # Auxiliary matrices that will ease the computation of averages and
    # covariances
    U = np.mat(np.ones((M,M))/M)
    I = np.mat(np.eye(M))

    # The ensemble is inflated (rho can be zero)
    Xb_pert = (1+rho)*Xb*(I-U)
    Xb = Xb_pert + Xb*U

    # Now, we choose from one of three methods

    # Stochastic Ensemble Kalman Filter
    if met=='SEnKF':

        # Create the ensemble in Y-space
        Yb = np.mat(np.empty((L,M)))
        Yb.fill(np.nan)

        # Map every ensemble member into observation space
        for jm in range(M):
            Yb[:,jm] = H*Xb[:,jm]

        # The matrix of perturbations
        Xb_pert = Xb*(I-U)
        Yb_pert = Yb*(I-U)

        # The Kalman gain matrix
        Khat = 1.0/(M-1)*Xb_pert*Yb_pert.T*pinv(1.0/(M-1)*Yb_pert*Yb_pert.T+R)

        # Fill Xa (the analysis matrix) member by member using
        # perturbed observations
        Xa = np.mat(np.empty((N,M)))
        Xa.fill(np.nan)
        for jm in range(M):
            yaux = y + np.real_if_close(sqrtm(R))*np.mat(np.random.randn(L,1))
            Xa[:,jm] = Xb[:,jm] + Khat*(yaux-Yb[:,jm])

    # Ensemble Transform Kalman Filter
    elif met=='ETKF':

        # Create the ensemble in Y-space
        Yb = np.mat(np.empty((L,M)))
        Yb.fill(np.nan)

        # Map every ensemble member into observation space
        for jm in range(M):
            Yb[:,jm] = H*Xb[:,jm]

        # Means and perts
        xb_bar = Xb*np.ones((M,1))/M
        Xb_pert = Xb*(I-U)
        yb_bar = Yb*np.ones((M,1))/M
        Yb_pert = Yb*(I-U)

        # The method
        Pa_ens = pinv((M-1)*np.eye(M)+Yb_pert.T*pinv(R)*Yb_pert)
        Wa = sqrtm((M-1)*Pa_ens) # matrix square root (symmetric)
        Wa = np.real_if_close(Wa)
        wa = Pa_ens*Yb_pert.T*pinv(R)*(y-yb_bar)
        Xa_pert = Xb_pert*Wa
        xa_bar = xb_bar + Xb_pert*wa
        Xa = Xa_pert + xa_bar*np.ones((1,M))

    return Xa


###############################################################################

def kfs_lor63_pe(x0guess,par0guess,t,tobs,y,H,R,rho,alpha,M,met):
 Nsteps = np.size(t)   
 N = np.size(x0guess)
 # For the true time
 tstep_truth = t[1]-t[0]
 # For the analysis (we assimilate everytime we get observations)
 tstep_obs = tobs[1]-tobs[0]
 # The ratio
 o2t = int(tstep_obs/tstep_truth+0.5)
 # Precreate the arrays for background and analysis
 Xb = np.empty((Nsteps,N,M)); Xb.fill(np.NaN)
 Xa = np.empty((Nsteps,N,M)); Xa.fill(np.NaN)
 Param_a = np.empty((len(tobs),len(par0guess),M)); Param_a.fill(np.nan)

 # For the original background ensemble
 # Two options: fixed and random
 back0 = 'fixed'
 #back0 = 'random'
 desv = 1

 # Fixed initial conditions for our ensemble (created ad hoc)
 if back0=='fixed':
  for j in range(N):
   Xb[0,j,:] = np.linspace(x0guess[j]-desv,x0guess[j]+desv,M)
  del j 
  for j in range(len(par0guess)):
   Param_a[0,j,:] = np.linspace(par0guess[j]-alpha,par0guess[j]+alpha,M)
  del j
  # Random initial conditions for our ensemble; let's perturb the
  # real x0 using errors with the magnitudes of R
 elif back0=='random':
  for j in range(M):
   Xb[0,:,j] = x0guess + desv*np.random.randn(N)
  del j
  for j in range(M):
   Param_a[0,:,j] = par0guess + alpha*np.random.randn(len(par0guess))
  del j
  
 # Since we don't have obs at t=0 the first analysis is the same as
 # background
 Xa[0,:,:] = Xb[0,:,:]

 # The following cycle contains evolution and assimilation for all
 # the time steps
 for j in range(len(tobs)-1):
  # First we evolve the ensemble members
  # Evolve from analysis!
  xold = Xa[j*o2t,:,:] # [N,M]
  paramold = Param_a[j,:,:] # [Nparam,M]
  # Time goes forward
  xnew = evolvemembers(xold,tstep_truth,o2t,paramold) # needs [N,M] arrays, gives [o2t+1,N,M]
  # The new background
  Xb[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  Xa[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  # The assimilation
  Xa_aux,Param_aux = enkfs_pe(Xb[(j+1)*o2t,:,:],paramold,y[j+1,:],H,R,rho,alpha,met)
  Xa[(j+1)*o2t,:,:] = Xa_aux # introduce the auxiliary variable
  Param_a[j+1,:,:] = Param_aux
  print ('t=',t[j*o2t])
 del j
 # The background and analysis mean
 x_b = np.mean(Xb,axis=2) # [t,N,M] -> [t,N]
 x_a = np.mean(Xa,axis=2) # [t,N,M] -> [t,N]
 param_a = np.mean(Param_a,axis=2) # [t,Nparam,M] -> [t,Nparam]
 return Xb,x_b,Xa,x_a,Param_a,param_a


## The EnKF algorithms
def enkfs_pe(Xb,paramb,y,H,R,rho,alpha,met):

    # General settings
    # The background information
    Xb = np.mat(Xb) # array -> matrix
    y = np.mat(y).T # array -> column vector
    paramb = np.mat(paramb) # array -> matrix

    # Number of state variables, ensemble members, observations and
    # parameters
    N,M = Xb.shape
    L,N = H.shape
    Nparam,M = paramb.shape

    # Auxiliary matrices that will ease the computation of averages and
    # covariances
    U = np.mat(np.ones((M,M))/M)
    I = np.mat(np.eye(M))

    # The ensemble is inflated (rho can be zero)
    Xb_pert = (1+rho)*Xb*(I-U)
    Xb = Xb_pert + Xb*U

    # The background values of the parameters are the same as the
    # analysis in the previous cycle.  We have to randomly perturb.
    paramb = paramb + alpha*np.random.randn(Nparam,M)

    # When doing parameter estimation, we have to extend the state
    # vector to include the parameters
    Xb = np.vstack((Xb,paramb))

    # We also need to modify the observation matrix
    H = np.hstack((H,np.zeros((L,Nparam))))

    # Now, we choose from one of three methods

    # Stochastic Ensemble Kalman Filter
    if met=='SEnKF':

        # Create the ensemble in Y-space
        Yb = np.mat(np.empty((L,M)))
        Yb.fill(np.nan)

        # Map every ensemble member into observation space
        for jm in range(M):
            Yb[:,jm] = H*Xb[:,jm]

        # The matrix of perturbations
        Xb_pert = Xb*(I-U)
        Yb_pert = Yb*(I-U)

        # The Kalman gain matrix
        Khat = 1.0/(M-1)*Xb_pert*Yb_pert.T*pinv(1.0/(M-1)*Yb_pert*Yb_pert.T+R)

        # Fill Xa (the analysis matrix) member by member using
        # perturbed observations.  Remember the matrix is extended
        # with the parameters.
        Xa = np.mat(np.empty((N+Nparam,M)))
        Xa.fill(np.nan)
        for jm in range(M):
            yaux = y + np.real_if_close(sqrtm(R))*np.mat(np.random.randn(L,1))
            Xa[:,jm] = Xb[:,jm] + Khat*(yaux-Yb[:,jm])

    # Ensemble Transform Kalman Filter
    elif met=='ETKF':

        # Create the ensemble in Y-space
        Yb = np.mat(np.empty((L,M)))
        Yb.fill(np.nan)

        # Map every ensemble member into observation space
        for jm in range(M):
            Yb[:,jm] = H*Xb[:,jm]

        # Means and perts
        xb_bar = Xb*np.ones((M,1))/M
        Xb_pert = Xb*(I-U)
        yb_bar = Yb*np.ones((M,1))/M
        Yb_pert = Yb*(I-U)

        # The method
        Pa_ens = pinv((M-1)*np.eye(M)+Yb_pert.T*pinv(R)*Yb_pert)
        Wa = sqrtm((M-1)*Pa_ens) # matrix square root (symmetric)
        Wa = np.real_if_close(Wa)
        wa = Pa_ens*Yb_pert.T*pinv(R)*(y-yb_bar)
        Xa_pert = Xb_pert*Wa
        xa_bar = xb_bar + Xb_pert*wa
        Xa = Xa_pert + xa_bar*np.ones((1,M))

    # Finally separate parameters and state variables
    parama = Xa[N:,:]
    Xa = Xa[:N,:]

    return Xa,parama































