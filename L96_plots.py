# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:10:17 2017
@author: jamezcua
"""
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
def plotL96(t,xt,N):
 plt.figure().suptitle('Lorenz 96 system - Truth')
 for i in range(N):
  plt.subplot(np.ceil(N/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.ylim([-20,20])
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)

 fig = plt.figure()
 fig.suptitle('Lorenz 96 system - Truth')
 ax = fig.add_subplot(111, projection='3d')
 jj,tt = np.meshgrid(np.arange(N),t)
 ax.plot_wireframe(jj,tt,xt,rstride=len(t))
 ax.set_xlabel('variable number')
 ax.set_ylabel('time')

 levs = np.linspace(-15,15,21)
 mycmap = plt.get_cmap('BrBG',21)
 plt.figure().suptitle('Lorenz 96 system - Truth')
 C = plt.contourf(np.arange(N),t,xt,cmap=mycmap,levels=levs)
 plt.contour(np.arange(N),t,xt,10,colors='k',linestyles='solid')
 plt.xlabel('variable number')
 plt.ylabel('time')
 plt.title('Hovmoller diagram')
 plt.colorbar(C,extend='both')    
   

##############################################################################
def plotL96obs(t,xt,Nx,tobs,y,observed_vars,exp_title):
 plt.figure().suptitle(exp_title)
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k',label='truth')
  if i in observed_vars:
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r',label='obs')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.ylim([-20,20])
  plt.xlim([0,t[-1]])
  if i==3:
   plt.legend()   
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)

        
#############################################################################        
def plotL96DA_kf(t,xt,tobs,y,Nx,observed_vars,Xb,xb,Xa,xa,exp_title):
 plt.figure().suptitle('Ensemble:'+exp_title)
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.plot(t,Xb[:,i,:],'--b')
  plt.plot(t,Xa[:,i,:],'--m')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)

 plt.figure().suptitle(exp_title)
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k',label='truth')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r',label='obs')
  plt.plot(t,xa[:,i],'m',label='analysis')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i    
 plt.legend()
 plt.subplots_adjust(wspace=0.7,hspace=0.3)


#############################################################################
def plotL96DA_var(t,xt,N,tobs,y,observed_vars,xb,xa,exp_title,anawin=None):
 plt.figure().suptitle(exp_title+'truth, observations, background, and analysis')
 for i in range(N):
  ax = plt.subplot(np.ceil(N/4.0),4,i+1)
  plt.plot(t,xt[:,i],'k',label='truth')
  if i in observed_vars:
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r',label='observations')
  plt.plot(t,xb[:,i],'b',label='background')
  plt.plot(t,xa[:,i],'m',label='analysis')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  if i==3:
   plt.legend()
  if anawin!=None:
   minor_ticks = np.arange(t[0], t[-1], (tobs[1]-tobs[0])*anawin)
   ax.set_xticks(minor_ticks, minor=True)
   ax.grid(which='minor', color='#CCCCCC', linestyle=':')
  plt.grid(True)

 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)
 

#####################################################
def plotL96DA_pf(t,xt,Nx,tobs,y,observed_vars,xpf,x_m):
 plt.figure().suptitle('Truth, Observations and Ensemble')
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xpf[:,i,:],'--m')
  plt.plot(t,xt[:,i],'-k',linewidth=2.0)
  plt.plot(t,x_m[:,i],'-m',linewidth=2)
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i 
 plt.subplots_adjust(wspace=0.7,hspace=0.3)


#############################################################################
def plotpar(Nparam,tobs,paramt_time,Parama,parama):
 plt.figure().suptitle('True Parameters and Estimated Parameters')
 for i in range(Nparam):
  plt.subplot(Nparam,1,i+1)
  plt.plot(tobs,paramt_time[:,i],'k')
  plt.plot(tobs,Parama[:,i,:],'--m')
  plt.plot(tobs,parama[:,i],'-m',linewidth=2)
  plt.ylabel('parameter['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i 
 plt.subplots_adjust(hspace=0.3)
        

#############################################################################
def plotRH(M,tobs,xt,xpf,rank):
 nbins = M+1
 plt.figure().suptitle('Rank histogram')
 for i in range(3):
  plt.subplot(1,3,i+1)
  plt.hist(rank[:,i],bins=nbins)
  plt.xlabel('x['+str(i)+']')
  plt.axis('tight')
 plt.subplots_adjust(hspace=0.3)
 

#############################################################################
def tileplotlocM(mat, lam,mycmap_out=None,vs_out=None,figout=None):
    if mycmap_out==None:
     mycmap = 'BrBG'
    else:
     mycmap = mycmap_out   
    if vs_out==None:
     vs=[-2,2]   
    else:
     vs = vs_out   
    N1,N2 = np.shape(mat)
    if figout==None:
     plt.figure()
    plt.pcolor(np.array(mat).T,edgecolors='k',cmap=mycmap,vmin=vs[0],vmax=vs[1])
    ymin,ymax = plt.ylim()
    plt.ylim(ymax,ymin)
    #plt.clim(-3,3)
    plt.colorbar()
    plt.title('Location matrix, lambda='+str(lam))
    plt.xlabel('variable number')
    plt.ylabel('variable number')
    plt.xticks(np.arange(0.5,N1+0.5),np.arange(N1))
    plt.yticks(np.arange(0.5,N2+0.5),np.arange(N2))
        
        

        