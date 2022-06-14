# Last revision: January 2020 for Python 3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common_misc import gen_obs, rmse_spread, createH, getBsimple_opts, getBcanadian
from common_plots import tileplotB, plotRMSP

from L96_model import lorenz96
from L96_var import var3d
from L96_plots import plotL96, plotL96obs, plotL96DA_var


##
'''
1. The Nature Run
   Let us perform a 'free' run of the model, which we will consider the truth
'''
model = 'L96'
x0 = None # true initial condition - we let the model spin from rest 
          # (x_n(t=0) = F, for all n )
tmax = 30 # The final time of the nature run simulation (model time-step is 0.025)
Nx = 12  # number of state variables, need Nx >=12

print('*** generating nature run ***')
t,xt = lorenz96(tmax,x0,Nx)

Nt = np.size(t)  # number of time-steps
plotL96(t,xt,Nx) # generate plots 

# %%
'''
2. Generate the background error covariance matrix B
   For details of Canadian quick method see Polavarapu et al, 2005
'''

sam_period = 4  # sampling period (number of model time-steps)

opt = 'simple' # method for generating B - 'simple' or 'canadian'

if opt=='simple':
 B,Bcorr = getBsimple_opts(model,Nx,sam_period)
elif opt=='canadian': 
 B,Bcorr = getBcanadian(model,Nx,sam_period)

# set plot properties
indt = range(0,Nt,sam_period)

examples = [1,2,3]; Nexamples = np.size(examples) # selction of variables to plot
cols_examples = ['k','g','c','b'] 
mycmap = 'BrBG'
mycmap_corr = 'PiYG' 

vs = [-4,4]
vs_corr = [-1,1]

# plot selected trajectories & values at sampling points
plt.figure()
plt.subplot(2,2,1) 
for j in range(Nexamples):
 plt.plot(t,xt[:,j],linestyle='-',color=cols_examples[j])
 plt.plot(t[indt],xt[indt,j],linestyle='',color=cols_examples[j],marker='o')
del j
plt.xlim([2,10])
plt.xlabel('time')
#plt.ylabel('selected grid points') 
plt.title('Lorenz 96 model trajectories for '+str()+' gridpoints' \
          +'\n sampling every '+str(sam_period)+' model time-steps')
plt.grid(True)

# plot background error covariance matrix
plt.subplot(2,2,2)
tileplotB(B,mycmap,vs,1)
plt.title('error covariance matrix')

# plot background error correlation matrix
plt.subplot(2,2,3)
tileplotB(Bcorr,mycmap_corr,vs_corr,1)
plt.title('error correlation matrix')

# plot background error standard deviations
indx = range(0,Nx,1)
ax = plt.subplot(2,2,4)
plt.scatter(indx,np.sqrt(np.diag(B)),c='r',s=50)
plt.xlabel('variable number')
plt.ylabel('std')
plt.title('background error std')
plt.xlim([-1,Nx])
minor_ticks = np.arange(1, Nx, 1)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
plt.grid(True)
plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)


# %%
'''
3. Generate a diagnosed B using iterative method of Yang et al, 2006
'''

''' 
3a. The observations
    Decide what variables to observe and then generate observations from the truth
'''
obsgrid = 'landsea' # options are 'all', '1010': observe every other variable, 
                # 'landsea': observe only half of domain
                    
period_obs = 2;  # number of time steps between observations
var_obs = 2.0    # observation error variance

exp_title = 'L96 system - ob freq: '+str(period_obs)+'dt, obs density: '+str(obsgrid)+', obs err var: '+str(format(var_obs, '.2f'))

print('*** generating the observations ***')
seed = 1 # initialise random seed
H, observed_vars = createH(obsgrid,model,Nx) # create observation operator
tobs,y,R = gen_obs(t,xt,period_obs,H,var_obs,seed) # generate observations and observation error 
                                                   # covariance matrix
plotL96obs(t,xt,Nx,tobs,y,observed_vars,exp_title) # plot observations vs. truth

# %%
''' 
3b. Iteration
'''
Ntobs,Ny = np.shape(y)  # dimension of observation array
ind_ana = range(period_obs,Nt,period_obs) # indices of observation times
Niter = 7 # number of iterations

Ball = np.empty((Nx,Nx,Niter+1)); Ball.fill(np.NaN) # 3D array containing matrix B at each iteration
Ball[:,:,0] = np.eye(Nx) # first guess is that B is the identity matrix

yall = np.empty((Ntobs,Ny,Niter)); yall.fill(np.NaN) # observation array
x0guess = np.empty((Nx,Niter)); x0guess.fill(np.NaN) # initial guess

for j in range(Niter):  # get yall, x0guess for each iteration
 tobs_aux,y_aux,R_aux = gen_obs(t,xt,period_obs,H,var_obs,j)
 yall[:,:,j] = y_aux
 np.random.seed(40+j)
 x0guess[:,j] = np.random.randn(Nx)
 del tobs_aux,y_aux,R_aux
del j    

# set weight parameter 
if obsgrid=='all':
 alpha = .5
elif obsgrid=='1010':
 alpha = 0.98
elif obsgrid=='landsea':   
 alpha = 0.95

for j in range(Niter): # iterate to find B
 xb,xa = var3d(x0guess[:,j],t,tobs,yall[:,:,0],H,Ball[:,:,j],R,model,Nx)
 xref = xt
 dxb = xb[ind_ana,:] - xref[ind_ana,:] 
 Baux = np.cov(dxb,rowvar=0)
 Ball[:,:,j+1] = alpha*Ball[:,:,j] + (1-alpha)*Baux
 del dxb, Baux
del j 
 
# set plot properties
mycmap = 'BrBG'
#vs = [-0.5,0.5]
vs = [-2,2]

# plot matrix B at each iteration
plt.figure()
for j in range(Niter+1):
 plt.subplot(3,4,1+j)
 tileplotB(np.squeeze(Ball[:,:,j]),mycmap,vs,1)
 plt.title('matrix B after '+str(j)+' iterations')
 plt.subplots_adjust(top=0.955,bottom=0.08,left=0.11,right=0.9,hspace=0.465,wspace=0.345)
del j

# plot background error standard deviation at each iteration
plt.figure()
for j in range(Niter):
 plt.scatter(indx,np.sqrt(np.diag(Ball[:,:,j])),s=20,c='grey')
del j
plt.scatter(indx,np.diag(np.sqrt(Ball[:,:,0])),s=50,c='k',label='initial guess')
plt.scatter(indx,np.sqrt(np.diag(Ball[:,:,-1])),s=50,c='b',label='final estimate')
plt.xlim([-1,Nx])
minor_ticks = np.arange(1, Nx, 1)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
# plt.grid(True)
plt.xlabel('variable number',fontsize=14)
plt.ylabel('error std',fontsize=14)
plt.title('estimated background error standard deviation over '+str(Niter)+' iterations')
plt.legend()












