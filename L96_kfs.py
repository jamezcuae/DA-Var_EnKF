# 2017 JA
import numpy as np
from scipy.linalg import pinv, sqrtm
from L96_model import lorenz96

def kfs_lor96(x0_t,t,tobs,y,H,R,rho,M,met,lam,loctype):
 """Data assimilation for Lorenz 1996 using Ensemble Kalman Filters.
 Inputs:  - x0_t, the real initial position
          - t, time array of the model (should be evenly spaced)
          - tobs, time array of the observations (should be evenly spaced 
            with a timestep that is a multiple of the model timestep)
          - y, the observations
          - H, observation matrix
          - R, the observational error covariance matrix
          - rho, inflation for P.  Notice we multiply (1+rho)*Xpert
            or P*(1+rho)^2.
          - M, the ensemble size
          - met, a string containing the method: 'SEnKF', 'ETKF'
          - lam, the localization radius in gridpoint units.  If None,
            it means no localization.
          - loctype, a string indicating the type of localization: 'GC'
            to use the Gaspari-Cohn function, 'cutoff' for a sharp cutoff

 Outputs: - Xb, the background ensemble 3D array [time,vars,members]
          - xb, background mean
          - Xa, the analysis ensemble 3D array [time,vars,members]
          - xa, analysis mean
          - locmatrix, localization matrix (or None if lam is None)"""

 # General settings
 # Number of observations and variables
 Nsteps = np.size(t)
 L,N = np.shape(H)
 # For the true time
 tstep_truth = t[1]-t[0]
 # For the analysis (we assimilate everytime we get observations)
 tstep_obs = tobs[1]-tobs[0]
 # The ratio
 o2t = int(tstep_obs/tstep_truth+0.5)
 # Precreate the arrays for background and analysis
 Xb = np.empty((Nsteps,N,M)); Xb.fill(np.nan)
 Xa = np.empty((Nsteps,N,M)); Xa.fill(np.nan)

 # For the original background ensemble
 # Two options: fixed and random
 back0 = 'fixed'
 #back0 = 'random'
 desv = 1.0

 # Fixed initial conditions for our ensemble (created ad hoc)
 if back0=='fixed':
  for j in range(N):
   Xb[0,j,:] = np.linspace(x0_t[j]-np.sqrt(desv), x0_t[j]+np.sqrt(desv), M)
  del j  
 # Random initial conditions for our ensemble
 elif back0=='random':
  for j in range(M):
   Xb[0,:,j] = x0_t + np.sqrt(desv)*np.random.randn(N)
  del j
  
 # Since we don't have obs at t=0 the first analysis is the same as
 # background
 Xa[0,:,:] = Xb[0,:,:]

 # Getting the R-localization weights
 if lam != None:
  locmatrix = getlocmat(N,L,H,lam,loctype)
 else:
  locmatrix = None

 # The following cycle contains evolution and assimilation for all time steps
 for j in range(len(tobs)-1):
  # Evolve from analysis!
  xold = Xa[j*o2t,:,:] # [N,M]
  # Time goes forward
  xnew = evolvemembers(xold,tstep_truth,o2t) # needs [N,M] arrays,
  # The new background
  Xb[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  Xa[j*o2t+1:(j+1)*o2t+1,:,:] = xnew[1:,:,:] # [o2t,N,M]
  # The assimilation
  Xa_aux = enkfs(Xb[(j+1)*o2t,:,:],y[j+1,:],H,R,rho,met,lam,locmatrix)
  Xa[(j+1)*o2t,:,:] = Xa_aux # introduce the auxiliary variable
  print('t=',t[j*o2t])
 del j
 # The background and analysis mean
 x_b = np.mean(Xb,axis=2) # [t,N,M] -> [t,N]
 x_a = np.mean(Xa,axis=2) # [t,N,M] -> [t,N]

 return Xb, x_b, Xa, x_a, locmatrix
 

############################################################################
def evolvemembers(xold,tstep_truth,o2t):
 """Evolving the members.
 Inputs:  - xold, a [N,M] array of initial conditions for the
            M members and N variables
          - tstep_truth, the time step used in the nature run
          - o2t, frequency of observations in time steps
 Outputs: - xnew, a [o2t+1,N,M] array with the evolved members"""

 t_anal = o2t*tstep_truth
 N,M = np.shape(xold)
 xnew = np.empty((o2t+1,N,M)); xnew.fill(np.nan)
 
 for j in range(M):
  taux,xaux = lorenz96(t_anal,xold[:,j],N) # [o2t+1,N]
  xnew[:,:,j] = xaux
 del j
 
 return xnew


##############################################################################
## The EnKF algorithms
def enkfs(Xb,y,H,R,rho,met,lam,locmatrix):
 """Performs the analysis using different EnKF methods.
 Inputs: - Xb, the ensemble background [N,M]
         - y, the observations [L]
         - H, the observation matrix [L,N]
         - R, the obs error covariance matrix [L,L]
         - rho, inflation for P.  Notice we multiply (1+rho)*Xpert
           or P*(1+rho)^2.
         - met, a string that indicated what method to use
         - lam, the localization radius
         - locmatrix, localization matrix
 Output: - Xa, the full analysis ensemble [N,M]"""

 # General settings
 # The background information
 Xb = np.mat(Xb) # array -> matrix
 y = np.mat(y).T # array -> column vector
 sqR = np.real_if_close(sqrtm(R))
 
 # Number of state variables, ensemble members and observations
 N,M = np.shape(Xb)
 L,N = np.shape(H)

 # Auxiliary matrices that will ease the computation of averages and
 # covariances
 U = np.mat(np.ones((M,M))/M)
 I = np.mat(np.eye(M))

 # The ensemble is inflated (rho can be zero)
 Xb_pert = (1+rho)*Xb*(I-U)
 Xb = Xb_pert + Xb*U

 # Create the ensemble in Y-space
 Yb = np.mat(np.empty((L,M))); Yb.fill(np.nan)

 # Map every ensemble member into observation space
 for jm in range(M):
  Yb[:,jm] = H*Xb[:,jm]
 del jm
        
 # The matrix of perturbations
 Xb_pert = Xb*(I-U)
 Yb_pert = Yb*(I-U)

 # Now, we choose from one of three methods
 # Stochastic Ensemble Kalman Filter
 if met=='SEnKF':
  if np.all(locmatrix) == None:
   # The Kalman gain matrix without localization
   Khat = 1.0/(M-1)*Xb_pert*Yb_pert.T * pinv(1.0/(M-1)*Yb_pert*Yb_pert.T+R)
  else:
   # The Kalman gain with localization
   Caux = np.mat(locmatrix.A * (Xb_pert*Yb_pert.T).A)
   Khat = 1.0/(M-1)*Caux * pinv(1.0/(M-1)*H*Caux+R)

  # Fill Xa (the analysis matrix) member by member using perturbed observations
  Xa = np.mat(np.empty((N,M))); Xa.fill(np.nan)
  for jm in range(M):
   yaux = y + sqR*np.mat(np.random.randn(L,1))
   Xa[:,jm] = Xb[:,jm] + Khat*(yaux-Yb[:,jm])
  del jm
        
 # Ensemble Transform Kalman Filter
 elif met=='ETKF':
  # Means
  xb_bar = Xb*np.ones((M,1))/M
  yb_bar = Yb*np.ones((M,1))/M
 
  if np.all(locmatrix) == None:
   # The method without localization (ETKF)
   Pa_ens = pinv((M-1)*np.eye(M)+Yb_pert.T*pinv(R)*Yb_pert)
   Wa = sqrtm((M-1)*Pa_ens) # matrix square root (symmetric)
   Wa = np.real_if_close(Wa)
   wa = Pa_ens*Yb_pert.T*pinv(R)*(y-yb_bar)
   Xa_pert = Xb_pert*Wa
   xa_bar = xb_bar + Xb_pert*wa
   Xa = Xa_pert + xa_bar*np.ones((1,M))
  else:   
   Xa = letkf(Xb_pert,xb_bar,Yb_pert,yb_bar,y,H,lam,locmatrix,R)
  
 return Xa


##############################################################################
## Localization functions
def getlocmat(N,L,H,lam,loctype):
    #To get the localization weights.
    indx = np.mat(range(N)).T
    indy = H*indx
    dist = np.mat(np.empty((N,L)))
    dist.fill(np.nan)

    # First obtain a matrix that indicates the distance (in grid points)
    # between state variables and observations
    for jrow in range(N):
        for jcol in range(L):
            dist[jrow,jcol] = np.amin([abs(indx[jrow]-indy[jcol]),\
                                       N-abs(indx[jrow]-indy[jcol])])
    # Now we create the localization matrix
    # If we want a sharp cuttof
    if loctype=='cutoff':
        locmatrix = 1.0*(dist<=lam)
    # If we want something smooth, we use the Gaspari-Cohn function
    elif loctype=='GC':
        locmatrix = np.empty_like(dist)
        locmatrix.fill(np.nan)
        for j in range(L):
            locmatrix[:,j] = gasparicohn(dist[:,j],lam)
    return locmatrix


def gasparicohn(z,lam):
    "The Gaspari-Cohn function."
    c = lam/np.sqrt(3.0/10)
    zn = abs(z)/c
    C0 = np.zeros_like(zn)
    for j in range(len(C0)):
        if zn[j]<=1:
            C0[j] = - 1.0/4*zn[j]**5 + 1.0/2*zn[j]**4 \
                    + 5.0/8*zn[j]**3 - 5.0/3*zn[j]**2 + 1
        if zn[j]>1 and zn[j]<=2:
            C0[j] = 1.0/12*zn[j]**5 - 1.0/2*zn[j]**4 \
                    + 5.0/8*zn[j]**3 + 5.0/3*zn[j]**2 \
                    - 5*zn[j] + 4 - 2.0/3*zn[j]**(-1)
    return C0
    
    
###############################################################################    
    

    
    