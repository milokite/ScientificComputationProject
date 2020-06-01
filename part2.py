"""Scientific Computation Project 3, part 2
Your CID here
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import os
plt.style.use('ggplot')

def microbes(phi,kappa,mu,L = 1024,Nx=1024,Nt=1201,T=600,display=False):
    """
    Question 2.2
    Simulate microbe competition model

    Input:
    phi,kappa,mu: model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of f when true

    Output:
    f,g: Nt x Nx arrays containing solution
    """

    #generate grid
    L = 1024
    x = np.linspace(0,L,Nx)
    dx = x[1]-x[0]
    dx2inv = 1/dx**2

    def RHS(y,t,k,r,phi,dx2inv):
        #RHS of model equations used by odeint

        n = y.size//2

        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        #Construct RHS
        R = f/(f+phi)
        dfdt = d2f + f[1:-1]*(1-f[1:-1])- R[1:-1]*g[1:-1]
        dgdt = d2g - r*k*g[1:-1] + k*R[1:-1]*g[1:-1]
        dy = np.zeros(2*n)
        dy[1:n-1] = dfdt
        dy[n+1:-1] = dgdt

        #Enforce boundary conditions
        a1,a2 = -4/3,-1/3
        dy[0] = a1*dy[1]+a2*dy[2]
        dy[n-1] = a1*dy[n-2]+a2*dy[n-3]
        dy[n] = a1*dy[n+1]+a2*dy[n+2]
        dy[-1] = a1*dy[-2]+a2*dy[-3]

        return dy


    #Steady states
    rho = mu/kappa
    F = rho*phi/(1-rho)
    G = (1-F)*(F+phi)
    y0 = np.zeros(2*Nx) #initialize signal
    y0[:Nx] = F
    y0[Nx:] = G + 0.01*np.cos(10*np.pi/L*x) + 0.01*np.cos(20*np.pi/L*x)

    t = np.linspace(0,T,Nt)

    #compute solution
    print("running simulation...")
    y = odeint(RHS,y0,t,args=(kappa,rho,phi,dx2inv),rtol=1e-6,atol=1e-6)
    f = y[:,:Nx]
    g = y[:,Nx:]
    print("finished simulation")
    if display:
        plt.figure()
        plt.contour(x,t,f)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of f')


    return f,g


def newdiff(f,h):
    """
    Question 2.1 i)
    Input:
        f: array whose 2nd derivative will be computed
        h: grid spacing
    Output:
        d2f: second derivative of f computed with compact fd scheme
    """

    d2f = np.zeros_like(f) #modify as needed

    #Coefficients for compact fd scheme
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140
    
    l = len(f)
    A = np.zeros((l,l))
    d = np.zeros(l)
    h = np.pi/100

    def f1(f,i,alpha,a,b,c,h):
        return (1/h**2)*((c/9)*(f[i+3]-2*f[i]+f[i-3])+(b/4)*(f[i+2]-2*f[i]+f[i-2])+a*(f[i+1]-2*f[i]+f[i-1]))


    A[0,:2] = [1,10]
    d[0] = (1/h**2)*((145/12)*f[0]-(76/3)*f[1]+(29/2)*f[2]-(4/3)*f[3]+(1/12)*f[4])
    A[-1,-2:] = [10,1]
    d[-1] = (1/h**2)*((145/12)*f[-1]-(76/3)*f[-2]+(29/2)*f[-3]-(4/3)*f[-4]+(1/12)*f[-5])

    for i in range(len(A)-4):
        A[i+1][i:i+3] = [alpha,1,alpha]
        d[i+1] = f1(f,i+1,alpha,a,b,c,h)

    A[-2,-3:] = [alpha,1,alpha]
    d[-2] = (1/h**2)*((c/9)*(f[1]-2*f[-2]+f[-5])+(b/4)*(f[0]-2*f[-2]+f[-4])+a*(f[-1]-2*f[-2]+f[-3]))
    A[-3,-4:-1] = [alpha,1,alpha]
    d[-3] = (1/h**2)*((c/9)*(f[0]-2*f[-3]+f[-6])+(b/4)*(f[-1]-2*f[-3]+f[-5])+a*(f[-2]-2*f[-3]+f[-4]))
    d2f = np.linalg.solve(A,d) 
    return d2f

def analyzefd():
    """
    Question 2.1 ii)
    Add input/output as needed

    """
    def findiff(f,h):
        fdash = [(f[i]-f[i+2])/(2*h) for i in range(len(f)-2)]
        fdash2 = [(fdash[i]-fdash[i+2])/(2*h) for i in range(len(fdash)-2)]
        return fdash2

    x = np.arange(0,2*np.pi,np.pi/100)
    f = [np.sin(2*x) for x in x]
    h = np.pi/100

    plt.plot(x[2:-2],findiff(f,h))
    plt.title('Second differential of sin(2x) using finite difference')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
    
    x = np.arange(0,2*np.pi,np.pi/100)
    f = [np.sin(2*x) for x in x]
    actdiff = [-4*np.sin(2*x) for x in np.arange(0,2*np.pi,np.pi/100)]
    diff = newdiff(f,1/100)
    fig = plt.figure()
    error = [abs(diff[i] - actdiff[i]) for i in range(len(diff))]
    plt.plot(x,np.log(error),'.')
    plt.title('Errors of newdiff method for sin(2x)')
    plt.ylabel('Log error')
    plt.xlabel('x')
    plt.show()
    
    x = np.arange(0,2*np.pi,np.pi/100)
    f = [np.sin(2*x) for x in x]
    h = np.pi/100
    actdiff = [-4*np.sin(2*x) for x in np.arange(0,2*np.pi,np.pi/100)][2:-2]
    diff = findiff(f,h)
    error = [abs(diff[i]-actdiff[i]) for i in range(len(diff))]
    fig = plt.figure()
    plt.plot(x[2:-2],np.log(error),'.')
    plt.title('Errors of finite difference method for sin(2x)')
    plt.ylabel('Log error')
    plt.xlabel('x')
    plt.show()
    
    hs = np.arange(1/10000,1/1000,1/10000)
    newdifft = np.zeros(len(hs))
    for i,h in enumerate(hs):
        x = np.arange(0,2*np.pi,2*np.pi*h)
        f = [np.sin(2*x) for x in x]
        start = time.time()
        newdiff(f,h)
        end = time.time()
        newdifft[i] = np.log(end-start)
    plt.plot(hs,newdifft,'.')
    plt.ylabel('log time')
    plt.xlabel('h')
    plt.title('Time taken to perform newdiff by gap between array')
    plt.show()
    
    hs = np.arange(1/10000,1/1000,1/10000)
    newdifft = np.zeros(len(hs))
    for i,h in enumerate(hs):
        x = np.arange(0,2*np.pi,2*np.pi*h)
        f = [np.sin(2*x) for x in x]
        start = time.time()
        findiff(f,h)
        end = time.time()
        newdifft[i] = np.log(end-start)
    plt.plot(hs,newdifft,'.')
    plt.ylabel('log time')
    plt.xlabel('h')
    plt.title('Time taken to perform findiff by gap between array')
    plt.show()
    return None


def dynamics():
    """
    Question 2.2
    Add input/output as needed

    """
       
    ##################################################
    #change to your working directory!               #
    os.chdir(r'C:\Users\milok\Desktop\Sci Comp CW3') #
    ##################################################
    
    data1 = np.load('data1.npy')
    data2 = np.load('data2.npy')
    data3 = np.load('data3.npy')
    A = np.load('data2.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')
    t=np.arange(0,20*np.pi/80,np.pi/80)
    
    phi=0.3
    L=1024
    Nx=1024
    fs = [0,0,0]
    gs = [0,0,0]
    for i,kappa in enumerate([1.5,1.7,2]):
        mu=0.4*kappa
        fs[i],gs[i] = microbes(phi,kappa,mu,L,Nx,display=True)
        fig = plt.figure(figsize=(12,6))
        plt.plot(fs[i][150:,0])
        plt.title('Population for x=0 with kappa={}'.format(kappa))
        plt.ylabel('Population')
        plt.xlabel('Time')
        plt.show()
    return None #modify as needed

if __name__=='__main__':
    analyzefd()
    
    dynamics()
    
