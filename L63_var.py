# 2017 JA
import numpy as np
from scipy.linalg import pinv
from scipy.optimize import fmin, fsolve
from L63_model import lorenz63

###############################################################################
def var3d(x0,t,tobs,y,H,B,R,model,N):
    """Data assimilation routine for both Lorenz 1963 & 1996 using 3DVar.
    Inputs:  - x0, the real initial conditions
             - t, time array of the model (should be evenly spaced)
             - tobs, time array of the observations (should be evenly
               spaced with a timestep that is a multiple of the model
               timestep)
             - y, the observations
             - H, observation matrix
             - B, the background error covariance matrix
             - R, the observational error covariance matrix
             - model, a string indicating the name of the model: 'lor63'
               or 'lor96'
             - N, the number of variables
    Outputs: - x_b, the background
             - x_a, the analysis"""

    # General settings
    # For the true time
    tstep_truth = t[1]-t[0]
    # For the analysis
    tstep_obs = tobs[1]-tobs[0]
    # The ratio
    o2t = int(tstep_obs/tstep_truth+0.5)

    # Precreate the arrays for background and analysis
    x_b = np.empty((len(t),N))
    x_b.fill(np.nan)
    x_a = np.empty((len(t),N))
    x_a.fill(np.nan)

    # For the original background ensemble let's start close from the truth
    orig_bgd = 'fixed'
    #orig_bgd = 'random'

    if orig_bgd=='fixed':
        indaux = np.arange(N)
        x0_aux = x0 + (-1)**indaux
    elif orig_bgd=='random':
        x0_aux = x0 + np.random.randn(N)

    # For the first instant b and a are equal
    x_b[0,:] = x0_aux
    x_a[0,:] = x0_aux

    # The following cycle contains evolution and assimilation
    for j in range(len(tobs)-1):
        yaux = y[j+1,:]

        # First compute background; our initial condition is the
        # forecast from the analysis at the previous observational
        # time
        xb0 = x_a[j*o2t,:]
        if model=='L63':
            taux,xbaux = lorenz63(xb0,o2t*tstep_truth)
        #elif model=='lor96':
        #    taux,xbaux = lorenz96(o2t*tstep_truth,xb0,N)

        x_b[j*o2t+1:(j+1)*o2t+1,:] = xbaux[1:,:]
        x_a[j*o2t+1:(j+1)*o2t+1,:] = xbaux[1:,:]

        xa_aux = one3dvar(xbaux[o2t,:],yaux,H,B,R)
        x_a[(j+1)*o2t,:] = xa_aux
        print('t =', t[o2t*(j+1)])

    return x_b,x_a


def one3dvar(xb,y,H,B,R):
    "The 3DVar algorithm for one assimilation window."
    y = np.mat(y).T      # array -> column vector
    xbvec = np.mat(xb).T # array -> column vector
    invB = pinv(B)
    invR = pinv(R)

    #opcmin = 'dirmin'
    opcmin = 'gradeq0'

    # The cost function in case the minimization is direct
    def costfun(xa):
        xa = np.mat(xa).T
        Jback = (xa-xbvec).T*invB*(xa-xbvec)
        Jobs = (y-H*xa).T*invR*(y-H*xa)
        J = Jback + Jobs
        return J

    # The gradient in case we prefer to find its roots
    def gradJ(xa):
        xa = np.mat(xa).T
        # The background term
        gJb = invB*(xa-xbvec)
        # The observational term
        gJo = -H.T*invR*(y-H*xa)
        gJ = gJb + gJo
        return gJ.A.flatten()

    if opcmin=='dirmin':
        xa = fmin(costfun,xb,xtol=1e-3,disp=False)
    elif opcmin=='gradeq0':
        xa = fsolve(gradJ,xb,xtol=1e-6)

    return xa



##############################################################################
def var4d(x0,t,tobs,anawin,y,H,B,R,model,N):
    """Data assimilation routine for both Lorenz 1963 & 1996 using 4DVar.
    Inputs:  - x0, the real initial conditions
             - t, time array of the model (should be evenly spaced)
             - tobs, time array of the observations (should be evenly
               spaced with a timestep that is a multiple of the model
               timestep)
             - anawin, length of the 4D assim window, expressed as
               number of future obs included
             - y, the observations
             - H, observation matrix
             - B, the background error covariance matrix
             - R, the observational error covariance matrix
             - model, a string indicating the name of the model: 'lor63'
               or 'lor96'
             - N, the number of variables
    Outputs: - x_b, the background
             - x_a, the analysis"""

    # General settings
    # For the true time
    tstep_truth = t[1]-t[0]
    # For the analysis
    tstep_obs = tobs[1]-tobs[0]
    # The ratio
    o2t = int(tstep_obs/tstep_truth+0.5)

    totana = (len(tobs)-1)/anawin

    # Precreate the arrays for background and analysis
    x_b = np.empty((len(t),N))
    x_b.fill(np.nan)
    x_a = np.empty((len(t),N))
    x_a.fill(np.nan)

    # For the original background ensemble let's start close from the truth
    orig_bgd = 'fixed'
    #orig_bgd = 'random'

    if orig_bgd=='fixed':
        indaux = np.arange(N)
        x0_aux = x0 + (-1)**indaux
    elif orig_bgd=='random':
        x0_aux = x0 + np.random.randn(N)

    # For the first instant b and a are equal
    x_b[0,:] = x0_aux
    x_a[0,:] = x0_aux

    # The following cycle contains evolution and assimilation
    for j in range(int(totana)):
        # Get the observations; these are distributed all over the
        # assimilation window
        yaux = y[anawin*j+1:anawin*(j+1)+1,:] # [anawin,L]

        # First compute background; our background is the forecast
        # from the analysis
        xb0 = x_a[j*anawin*o2t,:]
        if model=='L63':
            taux,xbaux = lorenz63(xb0,o2t*anawin*tstep_truth)
        #elif model=='lor96':
        #    taux,xbaux = lorenz96(o2t*anawin*tstep_truth,xb0,N)

        x_b[j*o2t*anawin:(j+1)*o2t*anawin+1,:] = xbaux
        xa0 = one4dvar(tstep_truth,o2t,anawin,xb0,yaux,H,B,R,model,N)

        if model=='L63':
            taux,xaaux = lorenz63(xa0,o2t*anawin*tstep_truth)
        #elif model=='lor96':
        #    taux,xaaux = lorenz96(o2t*anawin*tstep_truth,xa0,N)

        x_a[j*o2t*anawin:(j+1)*o2t*anawin+1,:] = xaaux
        print('t =', tobs[anawin*(j+1)])

    return x_b,x_a


def one4dvar(tstep_truth,o2t,anawin,x0b,y,H,B,R,model,N):
    "The 4DVar algorithm for one assimilation window."
    x0bvec = np.mat(x0b).T # array -> column vector
    y = np.mat(y).T        # array -> matrix [N, nobs in assim window]
    invB = pinv(B)
    invR = pinv(R)

    #opcmin = 'dirmin'
    opcmin = 'gradeq0'


    # The cost function in case the minimization is direct
    def costfun(xa):
        xavec = np.mat(xa).T
        Jback = (xavec-x0bvec).T*invB*(xavec-x0bvec)

        if model=='L63':
            taux,xaaux = lorenz63(xa,o2t*anawin*tstep_truth)
        #elif model=='lor96':
        #    taux,xaaux = lorenz96(o2t*anawin*tstep_truth,xa,N)

        indobs = range(o2t,anawin*o2t+1,o2t)
        xobs = xaaux[indobs,:]
        xobs = np.mat(xobs).T
        Jobs = np.empty(len(indobs))
        Jobs.fill(np.nan)
        for iJobs in range(len(indobs)):
            Jobs[iJobs] = (y[:,iJobs]-H*(xobs[:,iJobs])).T*invR \
                          *(y[:,iJobs]-H*(xobs[:,iJobs]))
        J = Jback + np.sum(Jobs)
        return J

    # The gradient in case we prefer to find its roots
    def gradJ(xa):
        xavec = np.mat(xa).T
        # The background term
        gJb = invB*(xavec-x0bvec)

        # For the model and observation error we need the TLM and adjoint
        # Choose the TLM/adjoint depending on the model
        if model=='L63':
            xaux,M = transmat_lor63(xa,o2t*anawin*tstep_truth)
        #elif model=='lor96':
        #    xaux,M = transmat_lor96(xa,o2t*anawin*tstep_truth,N)

        # The observation error term, evaluated at different times
        gJok = np.mat(np.empty((N,anawin)))
        gJok.fill(np.nan)
        for j in range(anawin):
            gJok[:,j] = -np.mat(M[:,:,o2t*(j+1)]).T*H.T*invR \
                        *(y[:,j]-H*np.mat(xaux[o2t*(j+1),:]).T)

        # Adding the two
        gJ = gJb + np.sum(gJok,axis=1)
        return gJ.A.flatten()

    if opcmin=='dirmin':
        xa = fmin(costfun,x0b,xtol=1e-3,disp=False)
    elif opcmin=='gradeq0':
        xa = fsolve(gradJ,x0b,xtol=1e-6)

    return xa




##############################################################################
def transmat_lor63(x0,tmax):
    """The transition matrix required for the TLM and the adjoint.

    Inputs:  - x0, an array containing the initial state
             - tmax, the maximum time.  Should be a multiple of the
               timestep 0.01.
    Outputs: - x, the evolved state [len(t) x 3]
               M, the transition matrix for small perturbations from
               the initial state to the evolved state [3 x 3 x len(t)]"""

    global N
    N = len(x0)
    tstep = 0.01
    taux = np.arange(0,tmax+tstep/2,tstep)
    # Settings for M
    M0 = np.eye(N)

    xaux = np.empty((len(taux),N))
    xaux.fill(np.nan)
    xaux[0,:] = x0
    M = np.empty((N,N,len(taux)))
    M.fill(np.nan)
    M[:,:,0] = M0

    for i in range(len(taux)-1): # for each time
        xaux[i+1,:],M[:,:,i+1] = integr(xaux[i,:],M[:,:,i],tstep)

    return xaux,M

def integr(xin,Min,tstep):
    global N

    # The integration is for both the model and the TLM at the same time!
    Varsold = np.concatenate((xin,Min.flatten()))

    k1 = faux(Varsold)
    k2 = faux(Varsold+1/2.0*tstep*k1)
    k3 = faux(Varsold+1/2.0*tstep*k2)
    k4 = faux(Varsold+tstep*k3)
    Varsnew = Varsold + 1/6.0*tstep*(k1+2*k2+2*k3+k4)

    xout = Varsnew[:N]
    Mout = np.reshape(Varsnew[N:],(N,N))

    return xout,Mout

def faux(Varsin):
    "The Lorenz 1963 model and its TLM."
    global N
    dx = f(Varsin[:N])
    F = tlm(Varsin[:N])
    dM = F*np.mat(np.reshape(Varsin[N:],(N,N)))
    dxaux = np.concatenate((dx,dM.A.flatten()))
    return dxaux

def f(x):
    sigma = 10.0;     b = 8/3.0;    r = 28.0
    # Initialize
    dx = np.empty_like(x)
    dx.fill(np.nan)
    # The fast
    dx[0] = sigma*(x[1]-x[0])
    dx[1] = x[0]*(r-x[2])-x[1]
    dx[2] = x[0]*x[1]-b*x[2]
    return dx

def tlm(x):
    s = 10.0;     b = 8/3.0;    r = 28.0
    F = np.mat([[ -s   ,    s,     0],\
                [r-x[2],   -1, -x[0]],\
                [  x[1], x[0],    -b]])
    return F





















