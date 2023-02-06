import numpy as np
#import jax.numpy as np

def Euler2fixedpt(dxdt, x_initial, Tmax=0.5, dt=0.001, xtol=1e-5, xmin=1e-0, Tmin=0.2, PLOT=False, inds=None, verbose=False, silent=False, Tfrac_CV=0):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = v(x) (where the function v(.) is called dxdt(.) in this code) 
    using the Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot
    verbose: if True print convergence criteria even if passed (function always prints out a warning if it doesn't converge).
    Tfrac_var: if not zero, maximal temporal CV (coeff. of variation) of state vector components, over the final
               Tfrac_CV fraction of Euler timesteps, is calculated and printed out.
               
    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

    if PLOT:
        if inds is None:
            N = x_initial.shape[0] # x_initial.size
            inds = [int(N/4), int(3*N/4)]
        xplot = x_initial[inds][:,None]

    Nmax = int(np.round(Tmax/dt))
    Nmin = int(np.round(Tmin/dt)) if Tmax > Tmin else int(Nmax/2)
    xvec = x_initial
    CONVG = False
    if Tfrac_CV > 0:
        xmean = zeros_like(xvec)
        xsqmean = zeros_like(xvec)
        Nsamp = 0
    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        if PLOT:
            #xplot = np.asarray([xplot, xvvec[inds]])
            xplot = np.hstack((xplot,xvec[inds][:,None]))
        
        if n >= (1-Tfrac_CV) * Nmax:
            xmean = xmean + xvec
            xsqmean = xsqmean + xvec**2
            Nsamp = Nsamp + 1

        if n > Nmin:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol:
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG and not silent: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))
        if Tfrac_CV > 0:
            xmean = xmean/Nsamp
            xvec_SD = np.sqrt(xsqmean/Nsamp - xmean**2)
            # CV = xvec_SD / xmean
            # CVmax = CV.max()
            CVmax = xvec_SD.max() / xmean.max()
            print(f"max(SD)/max(mean) of state vector in the final {Tfrac_CV:.2} fraction of Euler steps was {CVmax:.5}")

        #mybeep(.2,350)
        #beep

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(244459)
        plt.plot(np.arange(n+2)*dt, xplot.T, 'o-')

    return xvec, CONVG


# This is the input-output function (for mean-field spiking neurons) that you would use Max
def ricciardi_fI(mu, sigma, tau=0.01, Vt=20, Vr=0, tau_ref=0):
    """
     Calculates rate from the Ricciardi equation, with an error
     less than 10^{-5}. If the LIF neuron's voltage, V, satisfies between spikes

                  tau dV/dt = mu - V + sqrt(tau)*sigma*eta(t), 

     where eta is standard white noise, ricciardi calculates the firing rate.

     YA: this is based on Ran Darshan's Matlab version of 
         Carl Van Vreeswijk's Fortran (?) code.
         
     In: 
         mu and sigma: mean and SD of input (in mV)
             can be numpy arrays. If sigma is not scalar, no zero components are allowed.
             If either mu or sigma is scalar, output will have the shape of the other. 
             If both are arrays, both are flattened and output will be of
             shape (mu.size, sigma.size)
         tau, Vt, and Vr:
                 have to be scalar, and denote the membrane time-constant (in seconds), 
                 spiking threshold and reset voltages (in mV), respectively. 
     Out: firing rate (in Hz), with shape:
          if either mu or sigma is scalar, output will have the shape of the other. 
          if both mu and sigma are arrays, both are flattened and 
          rate.shape = (mu.size, sigma.size)
    """

    #assert np.isscalar(Vt) and np.isscalar(Vr) and np.isscalar(tau)
    # assert np.isscalar(mu) or np.isscalar(sigma)
    if Vt < Vr:
        raise ValueError("Threshold voltage lower than reset!")
    if np.any(sigma < 0.): 
        raise ValueError("Negative noise sigma!")

    if not np.isscalar(sigma):
        assert np.all(sigma > 0)  
        if not np.isscalar(mu):
            assert mu.shape == sigma.shape
#             mu = np.atleast_2d(mu.ravel()).T      # shape = (mu.size, 1)
#             sigma = np.atleast_2d(sigma.ravel())  # shape = (1, sigma.size)

    elif sigma == 0:
        return lif_regular(mu, tau, Vt, Vr)
        

    xp = (mu - Vr) / sigma
    xm = (mu - Vt) / sigma

#     if xm > 0:
#         rate = 1 / (f_ricci(xp) - f_ricci(xm))
#     elif xp > 0:
#         rate = 1 / ( f_ricci(xp) + np.exp(xm**2) * g_ricci(-xm) )
#     else:
#         rate = np.exp(-xm**2 - np.log(g_ricci(-xm) - np.exp(xp**2 - xm**2) * g_ricci(-xp)))

    rate = np.zeros_like(xm)
    rate[xm > 0] = 1 / (f_ricci(xp[xm > 0]) - f_ricci(xm[xm > 0]))
    inds = (xp > 0) & (xm <= 0)
    rate[inds] = 1 / ( f_ricci(xp[inds]) + np.exp(xm[inds]**2) * g_ricci(-xm[inds]) )
    rate[xp <= 0] = np.exp(-xm[xp <= 0]**2 - np.log(g_ricci(-xm[xp <= 0]) 
                         - np.exp(xp[xp <= 0]**2 - xm[xp <= 0]**2) * g_ricci(-xp[xp <= 0])))
    
    rate = 1 / (tau_ref + 1 / rate)
    return rate / tau


def lif_regular(mu, tau, Vt, Vr):
    rate = np.zeros_like(mu)
    rate[mu > Vt] = 1 / np.log((mu[mu > Vt] - Vr)/(mu[mu > Vt] - Vt))

    return rate / tau

def f_ricci(x):
    z = x / (1 + x)
    a = np.array([0.0, 
                  .22757881388024176, .77373949685442023, .32056016125642045, 
                  .32171431660633076, .62718906618071668, .93524391761244940, 
                  1.0616084849547165, .64290613877355551, .14805913578876898])

#    return np.log(2*x + 1) + a @ (-z)**np.arange(10)
    return np.log(2*x + 1) + (  a[1] *(-z)**1 + a[2] *(-z)**2 + a[3] *(-z)**3
                              + a[4] *(-z)**4 + a[5] *(-z)**5 + a[6] *(-z)**6
                              + a[7] *(-z)**7 + a[8] *(-z)**8 + a[9] *(-z)**9 )

def g_ricci(x):

    z = x / (2 + x)
    enum = (  3.5441754117462949 * z    - 7.0529131065835378 * z**2 
            - 56.532378057580381 * z**3 + 279.56761105465944 * z**4 
            - 520.37554849441472 * z**5 + 456.58245777026514 * z**6  
            - 155.73340457809226 * z**7 )
    
    denom = (1 - 4.1357968834226053 * z - 7.2984226138266743 * z**2 
            + 98.656602235468327 * z**3 - 334.20436223415163 * z**4 
            + 601.08633903294185 * z**5 - 599.58577549598340 * z**6 
            + 277.18420330693891 * z**7 - 16.445022798669722 * z**8)
    
    return enum / denom


Phi = ricciardi_fI

