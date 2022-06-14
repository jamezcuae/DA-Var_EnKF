# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:02:48 2017
@author: jamezcua
"""

import numpy as np

#############################################################################
def createH(obsgrid):
 Nx = 3   
 if obsgrid=='x':
  observed_vars = [0]
 if obsgrid=='y':
  observed_vars = [1]
 if obsgrid=='z':
  observed_vars = [2]
 if obsgrid=='xy':
  observed_vars = [0,1]
 if obsgrid=='xz':
  observed_vars = [0,2]
 if obsgrid=='yz':
  observed_vars = [1,2]
 if obsgrid=='xyz':
  observed_vars = [0,1,2]
 Ny = np.size(observed_vars)
 H = np.mat(np.zeros((Ny,Nx)))
 for i in range(Ny):
  H[i,observed_vars[i]] = 1.0
 del i
 return H, observed_vars


#############################################################################
def gen_obs(t,x_t,period_obs,H,var_obs,myseed=None):
    """This function generates [linear] observations from a nature run.
    Inputs:  - t, the time array of the truth [Nsteps]
             - x_t, the true run [Nsteps, N variables]
             - period_obs, observation period (in model t steps)
             - H, the observation matrix
             - var_obs, the observational error variance (diagonal R)
    Outputs: - tobs, the time array of the observations
             - y, the observations
             - R, the observational error covariance matrix"""
    # The size of the observation operator (matrix) gives us the number of
    # observations and state variables
    L,N = np.shape(H)

    # The time array for obs (and coincides with the assimilation window time)
    tobs = t[::period_obs]

    # Initialize observations array
    y = np.empty((len(tobs),L))
    y.fill(np.nan)

    # The covariance matrix (remember it is diagonal)
    R = var_obs*np.mat(np.eye(L))

    # The cycle that generates the observations
    if myseed==None:
     np.random.seed()
    elif myseed!=None:
     np.random.seed(myseed)   
    
    for j in range(1,len(tobs)): # at time 0 there are no obs
        x_aux = np.mat(x_t[period_obs*j,:]).T
        eps_aux = np.sqrt(R)*np.random.randn(L,1)
        y_aux = H*x_aux + eps_aux
        y[j,:] = y_aux.T

    return tobs,y,R


##############################################################################
def rmse_spread(xt,xmean,Xens,anawin):
    """Compute RMSE and spread.

    This function computes the RMSE of the background (or analysis) mean with
    respect to the true run, as well as the spread of the background (or
    analysis) ensemble.

    Inputs:  - xt, the true run of the model [length time, N variables]
             - xmean, the background or analysis mean
               [length time, N variables]
             - Xens, the background or analysis ensemble
               [length time, N variables, M ensemble members] or None if
               no ensemble
             - anawin, the analysis window length.  When assimilation
               occurs every time we observe then anawin = period_obs.
    Outputs: - rmse, root mean square error of xmean relative to xt
             - spread, spread of Xens.  Only returned if Xens != None."""

    la,N = np.shape(xt)

    # Select only the values at the time of assimilation
    ind = range(0,la,anawin)
    mse = np.mean((xt[ind,:]-xmean[ind,:])**2,axis=1)
    rmse = np.sqrt(mse)

    if Xens != None:
        spread = np.var(Xens[ind,:,:],ddof=1,axis=2)
        spread = np.mean(spread,axis=1)
        spread = np.sqrt(spread)
        return rmse,spread
    else:
        return rmse


##############################################################################
from L63_model import lorenz63

def getBsimple(model,N):
    """A very simple method to obtain the background error covariance.

    Obtained from a long run of a model.

    Inputs:  - model, the name of the model 'lor63' or 'lor96'
             - N, the number of variables
    Outputs: - B, the covariance matrix
             - Bcorr, the correlation matrix"""

    if model=='L63':
        total_steps = 10000
        tstep = 0.01
        tmax = tstep*total_steps
        x0 = np.array([-10,-10,25])
        t,xt = lorenz63(x0,tmax)
        samfreq = 16
        err2 = 2
#    elif model=='lor96':
#        total_steps = 5000
#        tstep = 0.025
#        tmax = tstep*total_steps
#        x0 = None
#        t,xt = lorenz96(tmax,x0,N)
#        samfreq = 2
#        err2 = 2

    # Precreate the matrix
    ind_sample = range(0,total_steps,samfreq)
    x_sample = xt[ind_sample,:]
    Bcorr = np.mat(np.corrcoef(x_sample,rowvar=0))

    B = np.mat(np.cov(x_sample,rowvar=0))
    alpha = err2/np.amax(np.diag(B))
    B = alpha*B

    return B,Bcorr
