# Easier version. 2017. JA
# Last reviewed January 2020 for python 3
# Small changes April 2021. PJS 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common_misc import gen_obs, rmse_spread, createH, getBsimple
from common_plots import tileplotB, plotRMSP

from L63_model import lorenz63
from L63_var import var3d, var4d
from L63_plots import plotL63, plotL63obs, plotL63DA_var

##
''' 
1. The Nature Run
   Let us perform a 'free' run of the model, which we will consider the truth
'''
model = 'L63'
x0 = [-10,-10,20] # true initial conditions
Nx = np.size(x0)  # number of state variables
tmax = 10  # The final time of the nature run simulation (model time-step is 0.01)

print('*** generating nature run, wait for integration ***')
paramtrue = [10.0,8/3.0,28.0]  # true parameters
t,xt = lorenz63(x0,tmax,paramtrue)  # run model
plotL63(t,xt) # plot truth trajectory
x0guess = [-11,-12,10]  # A guess to start from in our assimilation experiments

# %%
''' 2. The observations
    Decide what variables to observe and then generate observations from the truth
'''
obsgrid = 'xyz'  # options are 'x','y','z','xy','xz','yz','xyz'

period_obs = 10  # number of time steps between observations
var_obs = 1 # observation error variance   

exp_title = 'L63 system - obs freq: '+str(period_obs)+'dt, obs density: '+str(obsgrid)+', obs err var: '+str(format(var_obs, '.2f'))

print('*** generating the observations ***')
seed = 1
H,observed_vars = createH(obsgrid,model) # create observation operator
tobs,y,R = gen_obs(t,xt,period_obs,H,var_obs,seed) # generate observations and observation error 
                                                   # covariance matrix
plotL63obs(t,xt,tobs,y,observed_vars,exp_title) # plot observations vs. truth
    
# %%
'''  
3. Data Assimilation Using Variational Methods
   First we need to estimate the background error covariance matrix B
'''

Bpre,Bcorr = getBsimple(model,Nx) # create a climatological matrix (using very simple method)
tune = 1 # depends on the observational frequency
B = tune*Bpre

# set plot properties and then plot
mycmap = 'BrBG'  
vs = [-2,2]     
tileplotB(B,mycmap,vs)  


# %%
''' 
3.a. 3D-Var data assimilation
'''
print('*** performing 3D-Var assimilation ***')
xb,xa = var3d(x0guess,t,tobs,y,H,B,R,model,Nx)  # call 3D-Var routine
exp_title = 'Lorenz 63 model, 3D-Var: '
plotL63DA_var(t,xt,tobs,y,observed_vars,xb,xa,exp_title)  # plot output

'''
compute RMSE of background and analysis then plot
'''
rmse_step=1
rmseb = rmse_spread(xt,xb,None,rmse_step) 
rmsea = rmse_spread(xt,xa,None,rmse_step)

exp_title=('Lorenz 63 model, 3D-Var with ob freq: '+str(period_obs)+'dt, obs density: '
           +str(obsgrid)+', obs err var: '+str(format(var_obs, '.2f')))
plotRMSP(exp_title,t,rmseb,rmsea)

# %%
'''
3.b. 4D-Var data assimilation
'''
anawin=4 # length of assimilation window, expressed as multiple of obs frequency
print('*** performing 4D-Var assimilation ***')
xb,xa = var4d(x0guess,t,tobs,anawin,y,H,B,R,model,Nx) # call 4D-Var routine
exp_title = 'Lorenz 63 model, 4D-Var: '
plotL63DA_var(t,xt,tobs,y,observed_vars,xb,xa,exp_title,anawin) # plot output

'''
compute RMSE of background and analysis then plot
'''
rmse_step=1
rmseb = rmse_spread(xt,xb,None,rmse_step)
rmsea = rmse_spread(xt,xa,None,rmse_step)

exp_title=('Lorenz 63 model, 4D-Var with window length: '+str(anawin*period_obs)+'dt, ob freq: '
           +str(period_obs)+'dt, obs density: '
           +str(obsgrid)+', obs err var: '+str(format(var_obs, '.2f')))
plotRMSP(exp_title,t,rmseb,rmsea)





