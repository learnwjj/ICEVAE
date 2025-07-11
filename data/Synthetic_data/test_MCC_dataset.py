import math

import numpy as np
import torch
def dgp():
    Osize, Esize = 2000, 500
    betaZ=1
    L=np.random.randint(0, 7, size=(Osize+Esize, 1))
    U_cov = 2.5 * np.eye(2)
    X_cov = 1 * np.eye(2)
    U,X = [], []
    for i in range(Osize+Esize):
        u = np.random.multivariate_normal(
            mean= np.array([-1*L[i], 1.5*L[i]]).squeeze()
            , cov=U_cov,
            size=1)
        x = np.random.multivariate_normal(
            mean=( 0.4*np.dot(u,np.array([[2, 1],[1,1]])).squeeze()+0.6*np.dot(L[i],np.array([[-3, 3]])).squeeze() )
            , cov=X_cov, size=1)
        if i == 0:
            U = u
            X = x
        else:
            U, X = np.concatenate((U, u), axis=0), np.concatenate((X, x), axis=0)

    indices = np.random.permutation(Osize+Esize)

    E_indices = indices[:Esize]
    O_indices = indices[Esize:]
    LE = L[E_indices]
    UE = U[E_indices]
    XE = X[E_indices]
    LO = L[O_indices]
    UO = U[O_indices]
    XO = X[O_indices]

    TO = np.random.binomial(n=1, size=Osize, p=(1 / (1 + np.exp(-np.mean(np.concatenate((XO, betaZ * UO), 1), axis=1)))))
    TE = np.random.binomial(n=1, size=Esize, p=(1 / (1 + np.exp(-np.mean(XE, 1)))))


    print('sizeO,E=' + str(Osize) + str(',') + str(Esize))

    noiseSO, noiseSE = np.random.randn(Osize) * 0.7, np.random.randn(Esize) * 0.7
    SO_treated = 3 + np.mean(XO, axis=1) + 2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + noiseSO
    SO_control =                           2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + noiseSO
    SE_treated = 3 + np.mean(XE, axis=1) + 2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + noiseSE
    SE_control =                           2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + noiseSE

    noiseO, noiseE = np.random.randn(Osize) * 0.7, np.random.randn(Esize) * 0.7
    YO_treated = 4 + np.mean(XO, axis=1) + np.mean(XO, axis=1) + SO_treated +  betaZ * np.mean(UO, axis=1)+noiseO
    YO_control =     np.mean(XO, axis=1)                       + SO_control + betaZ * np.mean(UO, axis=1)+ noiseO
    YE_treated = 4 + np.mean(XE, axis=1) +np.mean(XE, axis=1) + SE_treated +  betaZ * np.mean(UE, axis=1)+ noiseE
    YE_control =     np.mean(XE, axis=1)                      + SE_control +  betaZ* np.mean(UE, axis=1)+noiseE

    SE, SO = np.where(TE==1, SE_treated, SE_control), np.where(TO==1, SO_treated, SO_control)
    YE, YO = np.where(TE==1, YE_treated, YE_control), np.where(TO==1, YO_treated, YO_control)

    ite_O, ite_E = YO_treated-YO_control, YE_treated-YE_control

    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    return torch.from_numpy(1.0*L), torch.from_numpy(U) , torch.from_numpy(X) , torch.from_numpy(T) , torch.from_numpy(S), torch.from_numpy(Y), torch.from_numpy(G), \
            torch.from_numpy(1.0*LO) ,torch.from_numpy(UO), torch.from_numpy(XO) , torch.from_numpy(TO), torch.from_numpy(SO),torch.from_numpy(YO), \
                              torch.from_numpy(1.0*LE) ,torch.from_numpy(UE), torch.from_numpy(XE) , torch.from_numpy(TE), torch.from_numpy(SE),torch.from_numpy(YE),\
            ite_O, ite_E