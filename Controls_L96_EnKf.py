# Easier version. 2017. JA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common_misc import gen_obs, rmse_spread, createH, getBsimple
from common_plots import tileplotB, plotRMSP
from L96_model import lorenz96
from L96_kfs import kfs_lor96
from L96_plots import plotL96, plotL96obs, plotL96DA_kf, plotRH, tileplotlocM


##
'''
 1. The Nature Run
 Let us perform a 'free' run of the model, which we will consider the truth
 The initial conditions
'''
model = 'L96'
x0 = None # let it spin from rest (x_n(t=0) = F, forall n )
tmax = 4 # The final time of the nature run simulation
Nx = 12 # number of state variables


print('***generating Nature run')
t,xt = lorenz96(tmax,x0,Nx) # Nx>=12
plotL96(t,xt,Nx)

# imperfect initial guess for our DA experiments
forc = 8.0; aux1 = forc*np.ones(Nx); aux2 = range(Nx); 
x0guess = aux1 + ((-1)*np.ones(Nx))**aux2
del aux1, aux2


# %%
'''
 2. The observations
 Decide what variables to observe
'''
obsgrid = '1010' # options are 'all', '1010': # Observe every other variable, 'landsea': # Observe left half
H, observed_vars = createH(obsgrid,model,Nx)
period_obs = 2# number of time steps between observtaions
var_obs = 2# error variance of the observations
# Generating the observations
seed = 1
print('***generating observations')
tobs,y,R = gen_obs(t,xt,period_obs,H,var_obs,seed)

exp_title = 'ob freq:'+str(period_obs)+', density:'+str(obsgrid)+', err var:'+str(var_obs)
plotL96obs(t,xt,Nx,tobs,y,observed_vars,exp_title)

#%%
'''
 3. Data assimilation using KFs   
# No LETKF since R-localisation is extremely slow without parallel implementation   
'''  
rho = 0.1 # inflation factor
M = 24 # ensemble size
lam = 1.0 #localization radius in gridpoint units.  If None,it means no localization.
loctype = 'GC'
met = 'SEnKF' 
exp_title_M=('ob freq:'+str(period_obs)+', density:'+str(obsgrid)+
             ', err var:'+str(var_obs)+', M='+str(M)+', lambda='+str(lam)+', rho='+str(rho))
Xb,xb,Xa,xa,locmatrix = kfs_lor96(x0guess,t,tobs,y,H,R,rho,M,met,lam,loctype)
plotL96DA_kf(t,xt,tobs,y,Nx,observed_vars,Xb,xb,Xa,xa,exp_title_M)

if np.all(locmatrix) != None:
 mycmap = 'BrBG'
 vs = [-2,2]
 tileplotlocM(locmatrix,lam,mycmap,vs)

rmse_step=1
rmseb,spreadb = rmse_spread(xt,xb,Xb,rmse_step)
rmsea,spreada = rmse_spread(xt,xa,Xa,rmse_step)
plotRMSP(exp_title_M,t,rmseb,rmsea,spreadb,spreada)


