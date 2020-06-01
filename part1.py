"""Scientific Computation Project 3, part 1
Your CID here 01201131
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy.integrate import odeint
from scipy.special import hankel1
from scipy.integrate import odeint
import time
plt.style.use('ggplot')

def hfield(r,th,h,levels=50):
    """Displays height field stored in 2D array, h,
    using polar grid data stored in 1D arrays r and th.
    Modify as needed.
    """
    thg,rg = np.meshgrid(th,r)
    xg = rg*np.cos(thg)
    yg = rg*np.sin(thg)
    plt.figure()
    plt.contourf(xg,yg,h,levels)
    plt.axis('equal')
    return None

def repair1(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data

    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac = 0.0
                    Asum = 0

                    for j in mlist[m]:
                        Bfac += B[n,j]**2
                        Rsum = 0
                        for k in range(p):
                            if k != n: Rsum += A[m,k]*B[k,j]
                        Asum += (R[m,j] - Rsum)*B[n,j]

                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m<p:
                    #Add code here to update B[m,n]
                    B[m,n]=None #modify
        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if z%10==0: print("z,dA,dB=",z,dA[z],dB[z])
            
    return A,B


def repair2(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data

    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Asum = 0
                    #took this out of the for loop
                    Bfac = sum(B[n,mlist[m]]**2)  
                    for j in mlist[m]:
                        #got rid of this for loop
                        Rsum = sum(np.delete(A[m,:],n)*np.delete(B[:,j],n))
                        Asum += (R[m,j] - Rsum)*B[n,j]
                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m < p:
                    #took this out of the for loop
                    Afac =sum(A[nlist[n],m]**2)
                    Bsum = 0
                    for i in nlist[n]:
                        #got rid of this for loop
                        Rsum = sum(np.delete(A[i,:],m)*np.delete(B[:,n],m))
                        Bsum += (R[i,n]-Rsum)*A[i,m]
                    B[m,n] = Bsum/(Afac+l)
        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        print("z,dA,dB=",z,dA[z],dB[z])
        
    return A,B


def outwave(r0):
    """
    Question 1.2i)
    Calculate outgoing wave solution at r=r0
    See code/comments below for futher details
        Input: r0, location at which to compute solution
        Output: B, wave equation solution at r=r0
    """
    import numpy as np
    A = np.load('data2.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')

    Nr,Ntheta,Nt = A.shape
    B = np.zeros((Ntheta,Nt))
    
    #plotting the height map for t=0
    hfield(r,th,data3[:,:,0])
    
    #defining potential function
    def hankel(k,m,w,r,c,tt):
        return k*(np.real(hankel1(m,w*r)*np.exp(-complex(0,1)*c*tt)+np.conj(hankel1(m,w*r)*np.exp(complex(0,1)*c*tt))))
    
    #attempting to fit the function over a range of possible parameter values
    i,j=0,1
    tt=t[i]
    m=th[j]
    c1=0
    c2=0
    min_err=1000
    iteration=0
    targ=A[:,j,i]
    for c in np.arange(-0.15,0.15,0.05):
        for k1 in np.arange(0,0.07,0.02):
            for w1 in np.arange(1,1.5,1.5):
                for k2 in np.arange(-0.5,0,0.1):
                    for w2 in np.arange(6,10,0.5):
                        iteration+=1
                        if iteration%200==0:
                            print('iteration={}'.format(iteration))
                        Y = [hankel(k1,m,w1,r,c1,tt)+hankel(k2,m,w2,r,c2,tt)+c for r in r]
                        err = np.abs(np.mean(np.abs(Y-targ)))
                        if err < min_err:
                            min_err = err
                            min_Y = Y
                            min_vars = k1,w1,k2,w2,c
    print('minimum variables k1,w1,k2,w2,c are',min_vars,'respectively and minimum error is',min_err)
    plt.plot(r,min_Y)
    plt.plot(r,targ)
    plt.show()
    
    rs = [hankel(min_vars[1],m,min_vars[1],r,0,tt)+hankel(min_vars[2],m,min_vars[3],r,0,tt)+min_vars[4] for r in r]
    B = np.zeros((len(th),len(r)))
    for i in range(len(th)):
        m = th[i]
        B[i] = [hankel(min_vars[1],m,min_vars[1],r,0.1,tt)+hankel(min_vars[2],m,min_vars[3],r,-0.1,tt)+min_vars[4] for r in r]
    hfield(r,th,B.T)    
    return B.T

def analyze1():
    """
    Question 1.2ii)
    Add input/output as needed

    """
    data3_0 = data3[:,289//8,:]
    data3_1 = data3[:,289*3//8,:]
    data3_2 = data3[:,289*5//8,:]
    
    fig, axs = plt.subplots(1,3,figsize=(10,6))
    fig.suptitle('Height dynamics for island at various angles',fontsize=16)
    for i,j in enumerate([1,3,5]):
        axs[i].imshow(data3[:,289*j//8,:])
        axs[i].set_title('θ={}pi/8'.format(j))
        axs[i].set_ylabel('Time')
        axs[i].set_xlabel('Distance')
    plt.show()
    
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle('Height dynamics for island at t=0',fontsize=18)
    te = np.arange(119)
    for i,j in enumerate([1,3,5]):
        axs[i].plot(te,data3[0,289*j//8,:])
        axs[i].set_title('θ={}pi/8'.format(j))
        axs[i].set_ylabel('Height')
        axs[i].set_xlabel('Distance')
    plt.show()
    
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle('Height dynamics for island at radius=1',fontsize=18)
    te = np.arange(119)
    for i,j in enumerate([1,3,5]):
        axs[i].plot(r,data3[:,289*j//8,0])
        axs[i].set_title('θ={}pi/8'.format(j))
        axs[i].set_ylabel('Height')
        axs[i].set_xlabel('Time')
    plt.show()
    return None #modify as needed




def reduce(H,p=20):
    """
    Question 1.3: Construct one or more arrays from H
    that can be used by reconstruct
    Input:
        H: 3-D data array
        inputs: can be used to provide other input as needed
    Output:
        arrays: a tuple containing the arrays produced from H
    """
    M,N,T=H.shape
    H_2 = np.zeros((M,p,T))
    Us = np.zeros((N,T))
    var = 0
    total_var = 0
    G_2 = np.zeros((M,N,T))
    for i in range(T):
        X = H[:,:,i]
        mean = np.outer(np.ones((M,1)),X.mean(axis=0))
        X2 = X - mean
        U,S,VT = np.linalg.svd(X2.T)
        G = np.dot(U.T,X2.T)
        G_2[:,:,i] = G.T
        H_2[:,:,i] = G[:p].T
        Us[:,i] = U[0]
        var+=sum(S[:p])
        total_var+=sum(S)
    print('The proportion of variance accounted for is {}% using only {}% of the matrix'
          .format(np.round(100*var/total_var,3),np.round(100*p/289,3)))
    
    return H_2,Us,G_2



def reconstruct(G,U,G_2,inputs=()):
    """
    Question 1.3: Generate matrix with same shape as H (see reduce above)
    that has some meaningful correspondence to H
    Input:
        arrays: tuple generated by reduce
        inputs: can be used to provide other input as needed
    Output:
        Hnew: a numpy array with the same shape as H
    """
    M,p,T = G.shape
    N,T = U.shape
    Hnew = np.zeros((M,N,T))

    for i in range(T):
        Hnew = np.outer(U[:,i],G_2[:,0,i]).T
        
    return Hnew

if __name__=='__main__':
    
    #loading in data
    os.chdir(r'C:\Users\milok\Desktop\Sci Comp CW3')
    data1 = np.load('data1.npy')
    data2 = np.load('data2.npy')
    data3 = np.load('data3.npy')
    A = np.load('data2.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')
    t=np.arange(0,20*np.pi/80,np.pi/80)
    
    A,B = repair2(data1,10,niter=15,l=50)
    R = np.matmul(A,B)
    hfield(r,th,R)
    
    outwave(0)
    
    analyze1()
    
    reduced_H,Us,G_2 = reduce(data3)
    
    Hnew = reconstruct(reduced_H,Us,G_2)
