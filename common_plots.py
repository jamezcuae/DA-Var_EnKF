# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:10:17 2017
@author: jamezcua
"""
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
def plotL63(t,xt):
 plt.figure().suptitle('Truth')
 for i in range(3):
  plt.subplot(3,1,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
  plt.subplots_adjust(hspace=0.3)

 fig = plt.figure()
 fig.suptitle('Truth')
 ax = fig.add_subplot(111, projection='3d')
 ax.plot(xt[:,0],xt[:,1],xt[:,2],'k')
 ax.set_xlabel('x[0]')
 ax.set_ylabel('x[1]')
 ax.set_zlabel('x[2]')
 ax.grid(True)


##############################################################################
def plotL63obs(t,xt,tobs,y,observed_vars):
 plt.figure().suptitle('Truth and Observations')
 for i in range(3):
  plt.subplot(3,1,i+1)
  plt.plot(t,xt[:,i],'k')
  if i in observed_vars:
 #  plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
  plt.subplots_adjust(hspace=0.3)
        
        
#############################################################################        
def plotL63DA_kf(t,xt,tobs,y,observed_vars,Xb,xb,Xa,xa):
 plt.figure().suptitle('Truth, Observations, Background Ensemble and Analysis Ensemble')
 for i in range(3):
  plt.subplot(3,1,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.plot(t,Xb[:,i,:],'--b')
  plt.plot(t,Xa[:,i,:],'--m')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
  plt.subplots_adjust(hspace=0.3)
 del i
 
 plt.figure().suptitle('Truth, Observations, Background and Analysis Mean')
 for i in range(3):
  plt.subplot(3,1,i+1)
  plt.plot(t,xt[:,i],'k')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.plot(t,xb[:,i],'b')
  plt.plot(t,xa[:,i],'m')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
  plt.subplots_adjust(hspace=0.3)
 del i 


#############################################################################
def plotL63DA_var(t,xt,tobs,y,observed_vars,xb,xa):
 plt.figure().suptitle('Truth, Observations, Background, and Analysis')
 for i in range(3):
  plt.subplot(3,1,i+1)
  plt.plot(t,xt[:,i],'k')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.plot(t,xb[:,i],'b')
  plt.plot(t,xa[:,i],'m')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i           
 plt.subplots_adjust(hspace=0.3)


#####################################################
def plotL63DA_pf(t,xt,tobs,y,observed_vars,xpf,x_m):
 plt.figure().suptitle('Truth, Observations and Ensemble')
 for i in range(3):
  plt.subplot(3,1,i+1)
  plt.plot(t,xpf[:,i,:],'--m')
  plt.plot(t,xt[:,i],'-k',linewidth=2.0)
  plt.plot(t,x_m[:,i],'-m',linewidth=2)
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
  plt.subplots_adjust(hspace=0.3)


############################################################################
def plotRMSP(exp_title,t,rmseb=None,rmsea=None,spreadb=None,spreada=None):
 plt.figure()
 plt.subplot(2,1,1)
 if np.all(rmseb)!=None:
  plt.plot(t,rmseb,'b',label='background')
 plt.plot(t,rmsea,'m',label='analysis')
 plt.legend()
 plt.ylabel('RMSE')
 plt.xlabel('time')
 plt.title(exp_title)
 plt.grid(True)

 if np.all(spreadb)!=None:
  plt.subplot(2,2,3)
  if np.all(rmseb)!=None:
   plt.plot(t,rmseb,'b',label='RMSE')
  plt.plot(t,spreadb,'--k',label='spread')
  plt.legend()
  plt.title('background')
  plt.xlabel('time')
  plt.grid(True)

 if np.all(spreada)!=None:
  plt.subplot(2,2,4)
  plt.plot(t,rmsea,'m',label='RMSE')
  plt.plot(t,spreada,'--k',label='spread')
  plt.legend()
  plt.title('analysis')
  plt.xlabel('time')
  plt.grid(True)

 plt.subplots_adjust(hspace=0.25)


#############################################################################
def tileplotB(mat, mycmap_out=None,vs_out=None,figout=None):
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
    plt.colorbar(extend='both')
    plt.title('matrix B')
    plt.xlabel('variable number')
    plt.ylabel('variable number')
    plt.xticks(np.arange(0.5,N1+0.5),np.arange(N1))
    plt.yticks(np.arange(0.5,N2+0.5),np.arange(N2))


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
        
        
        
