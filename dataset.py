# This is a sample Python script.
import math
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch

from matplotlib import style
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KernelDensity
from scipy import integrate



class DGP():

    def __int__(self):
        return

    def normalization(self, X):
        x=X.clone()
        mout=np.empty((1,x.shape[1] ))
        vout=np.empty((1, x.shape[1]))
        for i in range( x.shape[1] ):
            mout[0,i],vout[0,i]=x[:,i].mean(), x[:,i].std()
            m, v = x[:,i].mean(), x[:,i].std()
            x[:,i] = (x[:,i] - m)/ v
        return x,mout,vout

    def recover_Normalization(self,X,mout,vout):
        x = X.clone()
        for i in range(x.shape[1]):
            m, v = mout[0, i], vout[0, i]
            x[:, i] = x[:, i]*v + m
            # print("v",v, "m",m)
        return x, mout, vout

    def normalization_exp(self,X,mean,var):
        x = X.clone()
        for i in range(x.shape[1]):
            m, v = mean[0, i], var[0, i]
            x[:, i] = (x[:,i] - m)/ v
            # print("v",v, "m",m)
        return x

    def  normalization_exp_and_obs(self,X_exp,X_obs):
        x_exp=X_exp.clone()
        x_obs=X_obs.clone()
        x_exp,mout_exp,vout_exp=self.normalization(X_exp)
        x_obs, mout_obs, vout_obs = self.normalization(X_obs)
        mout_total=(x_exp.shape[0]*mout_exp+x_obs.shape[0]*mout_obs)/(x_exp.shape[0]+x_obs.shape[0])
        vout_total=(x_exp.shape[0]*(vout_exp*vout_exp+(mout_exp-mout_total)*(mout_exp-mout_total))+\
                    x_obs.shape[0]*(vout_obs*vout_obs+(mout_obs-mout_total)*(mout_obs-mout_total)))\
                   /(x_exp.shape[0]+x_obs.shape[0])
        vout_total=np.sqrt(vout_total)
        #重新归一化
        for i in range( X_exp.shape[1] ):
            m, v = mout_total[0, i], vout_total[0, i]
            X_exp[:,i] = (X_exp[:,i] - m)/ v

        for j in range(X_obs.shape[1]):
            m, v = mout_total[0, j], vout_total[0, j]
            X_obs[:, j] = (X_obs[:, j] - m) / v

        return X_exp,X_obs,mout_total,vout_total



    def Generate_S(self,Obsdata_size_input,x,t,z,noiseS):
        Obsdata_size = Obsdata_size_input
        S = 1 * np.mean(x, axis=1) + 3 * t + 0 * np.mean(x, axis=1) * t + 4 * np.mean(z, axis=1) * t + np.mean(z, axis=1) + noiseS

        return S

    def Generate_Y(self,Obsdata_size_input,x,t,z,s,noiseY):
        Obsdata_size = Obsdata_size_input
        Y = 4 * t + (0 * np.mean(x, axis=1) + 3 * np.mean(z, axis=1)) * t + 1 * np.mean(x, axis=1) + np.mean(z,axis=1) + s + noiseY

        return Y

    def GenerateDataset_STDmethod(self, Obsdata_size_input):
        noise_coef = 10
        Obsdata_size = Obsdata_size_input

        U = np.random.randint(0, 5, size=(Obsdata_size, 1))

        Z, X = [], []
        w_z2x = np.random.randn(2, 2)
        for i in range(Obsdata_size):
            z = np.random.multivariate_normal(mean=[U[i, 0], U[i, 0]**3], cov=[[1, 0], [0, 1]],
                                              size=1)
            x = np.matmul(z,w_z2x) + np.random.multivariate_normal(mean=[0,0], cov=[[1, 0], [0, 1]], size=1) * noise_coef

            if i == 0:
                Z = z
                X = x
            else:
                Z, X = np.concatenate((Z, z), axis=0), np.concatenate((X, x), axis=0)


        p_T=(1 / (1 + np.exp((-0.01) * np.mean(np.concatenate((X, Z), 1), axis=1))))
        print("obs-number>0.9", np.sum(p_T > 0.9))
        print("obs-number<0.1", np.sum(p_T < 0.1))
        T = np.random.binomial(n=1, size=Obsdata_size,p=p_T)

        T0 = np.zeros_like(T)
        T1 = np.ones_like(T)

        noiseS = np.random.randn(Obsdata_size) * 0.7

        noiseY = np.random.randn(Obsdata_size) * 0.7


        S = self.Generate_S(Obsdata_size,X,T,Z,noiseS)

        Y = self.Generate_Y(Obsdata_size,X,T,Z,S,noiseY)


        S1 = self.Generate_S(Obsdata_size,X,T1,Z,noiseS)
        S0 = self.Generate_S(Obsdata_size,X,T0,Z,noiseS)

        Y1 = self.Generate_Y(Obsdata_size,X,T1,Z,S1,noiseY)
        Y0 = self.Generate_Y(Obsdata_size,X,T0,Z,S0,noiseY)

        Real_Ite_List=Y1-Y0
        RealATE = np.mean(Y1-Y0)




        return torch.from_numpy(1.0 * U), torch.from_numpy(Z), torch.from_numpy(X), torch.from_numpy(
            T), torch.from_numpy(
            S), torch.from_numpy(Y),Real_Ite_List,RealATE





    def Generate_Exp__S(self, Expdata_size_input, x, t, z, noiseS):
        Expdata_size = Expdata_size_input
        S = 1 * np.mean(x, axis=1) + 3 * t + 0 * np.mean(x, axis=1) * t + 4 * np.mean(z, axis=1) * t + np.mean(z, axis=1) + noiseS

        return S

    def Generate_Exp_Y(self, Expdata_size_input, x, t, z, s, noiseY):
        Expdata_size = Expdata_size_input
        Y = 4 * t + (0 * np.mean(x, axis=1) + 3 * np.mean(z, axis=1)) * t + 1 * np.mean(x, axis=1) + np.mean(z, axis=1) + s + noiseY

        return Y

    def Generate_Exp_Dataset_STDmethod(self, Expdata_size_input):
        Expdata_size = Expdata_size_input

        U = np.random.randint(0, 5, size=(Expdata_size, 1))

        Z, X = [], []
        w_z2x = np.random.randn(2, 2)
        for i in range(Expdata_size):
            z = np.random.multivariate_normal(mean=[U[i, 0], U[i, 0] ** 3], cov=[[1, 0], [0, 1]],
                                                  size=1)

            x = z[0, 0] + np.random.uniform(low=0, high=100, size=2)
            x = x[:, None].transpose()

            if i == 0:
                Z = z
                X = x
            else:
                Z, X = np.concatenate((Z, z), axis=0), np.concatenate((X, x), axis=0)


        T = np.random.binomial(n=1, size=Expdata_size,
                                   p=(1 / (1 + np.exp((-0.01) * np.mean(X, axis=1)))))

        T0 = np.zeros_like(T)
        T1 = np.ones_like(T)

        noiseS = np.random.randn(Expdata_size) * 0.7

        noiseY = np.random.randn(Expdata_size) * 0.7


        S = self.Generate_S(Expdata_size, X, T, Z, noiseS)

        Y = self.Generate_Y(Expdata_size, X, T, Z, S, noiseY)


        S1 = self.Generate_S(Expdata_size, X, T1, Z, noiseS)
        S0 = self.Generate_S(Expdata_size, X, T0, Z, noiseS)

        Y1 = self.Generate_Y(Expdata_size, X, T1, Z, S1, noiseY)
        Y0 = self.Generate_Y(Expdata_size, X, T0, Z, S0, noiseY)

        Real_Ite_List = Y1 - Y0
        RealATE = np.mean(Y1 - Y0)




        return torch.from_numpy(1.0 * U), torch.from_numpy(Z), torch.from_numpy(X), torch.from_numpy(
                T), torch.from_numpy(
                S), torch.from_numpy(Y), Real_Ite_List, RealATE




    def naiveATE(self,x,z):

        return torch.mean(7+2*torch.mean(x, axis=1))

    def naiveATEmethodFrom(self,x,t,y):
        x=x.numpy()
        t=t.numpy()
        y=y.numpy()
        X=np.column_stack((x,t))
        regression_model = LinearRegression()
        print(y.shape)
        print(X.shape)
        regression_model.fit(X, y)
        c_yt_direct = regression_model.coef_[-1]

        return c_yt_direct




