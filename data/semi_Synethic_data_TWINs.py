
import math
import os
import random
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


class newDGP():
    def __int__(self):
        return


    def Generate_Obs_S_Y(self, Obsdata_size_input, x, t, z, W_yj, beta_z2t,
                         noiseS_Y):
        Obsdata_size = Obsdata_size_input

        part1 = np.sum(W_yj[5:10] * x[:, :5] , axis=1)

        part2 = 0.25*np.sum(W_yj[10:] * x[:, 5:10], axis=1)

        part3 = np.sum(W_yj[:5] * np.cos(z[:, :5]), axis=1)
        S = []
        S_out = []
        Y_out = []
        for i in range(14):

            if i == 0:
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + noiseS_Y[:, i]
            elif i == 1:
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + 0.25 * S + noiseS_Y[:, i]
            else:
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + 0.25 * np.mean(S[:, :i], axis=1) + noiseS_Y[ :,i]


            if i == 0:
                S = S_i
                S_out = S_i
            elif i > 0 and i < 7:
                S = np.column_stack((S, S_i))
                S_out = np.column_stack((S_out, S_i))
            elif i >= 7 and i < 13:
                S = np.column_stack((S, S_i))
            else:
                Y_out = S_i

        return S_out, Y_out


    def Generate_Exp_S_Y(self, Obsdata_size_input, x, t, z, W_yj, beta_z2t, noiseS_Y):
        Obsdata_size = Obsdata_size_input

        part1 = np.sum(W_yj[5:10] * x[:, :5] , axis=1)

        part2 = 0.25*np.sum(W_yj[10:] * x[:, 5:10], axis=1)

        part3 = np.sum(W_yj[:5] * np.cos(z[:, :5]), axis=1)
        S = []
        S_out = []
        Y_out = []
        for i in range(14):

            if i == 0:
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + noiseS_Y[:, i]
            elif i == 1:
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + 0.25 * S + noiseS_Y[:, i]
            else:
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + 0.25 * np.mean(S[:, :i],axis=1) + noiseS_Y[:,i]

            if i == 0:
                S = S_i
                S_out = S_i
            elif i > 0 and i < 7:
                S = np.column_stack((S, S_i))
                S_out = np.column_stack((S_out, S_i))
            elif i >= 7 and i < 13:
                S = np.column_stack((S, S_i))
            else:
                Y_out = S_i

        return S_out, Y_out


    def GenerateDataset_STDmethod(self, dir="../data/TWINS/",obs_ratio=0.6, seed=0):

        print(os.getcwd())

        data = pd.read_csv(dir + "twins_data_cleaned.csv")
        obs_len = int(len(data) * obs_ratio)
        exp_len = int(len(data) * (1-obs_ratio))
        random.seed(seed)
        np.random.seed(seed)
        idx = list(range(len(data)))
        random.shuffle(idx)
        data_obs = data.loc[idx[:obs_len]]
        data_exp = data.loc[idx[obs_len:]]

        U_obs = data_obs["dmeduc"].values[:,None]
        Z_obs = data_obs[["drink", "cigar", "resstatb", "adequacy", "dmar"]].values
        X_obs = data_obs[["nprevist", "cardiac", "lung", "dtotord", "diabetes", "chyper", "phyper", "wtgain", "dmage",
                          "gestat"]].values

        U_exp = data_exp["dmeduc"].values[:,None]
        Z_exp = data_exp[["drink", "cigar", "resstatb", "adequacy", "dmar"]].values
        X_exp = data_exp[["nprevist", "cardiac", "lung", "dtotord", "diabetes", "chyper", "phyper", "wtgain", "dmage",
                          "gestat"]].values

        Obsdata_size=obs_len
        Expdata_size=exp_len



        W_tj = np.random.uniform(0, 0.5, 10)
        W_yj = np.random.uniform(0.5, 1, 15)
        beta_z2t = 0.5

        T_values = [0, 1]
        weight_x2t = np.random.uniform(low=-0.5, high=0.5, size=(10, 1))
        weight_z2t = np.random.uniform(low=-0.5, high=0.5, size=(5, 1))
        # generate T_obs
        T_obs_scores = np.exp(
            (beta_z2t * np.dot(Z_obs, weight_z2t) + np.dot(X_obs, weight_x2t)).squeeze() / (np.mean(X_obs, axis=1) \
                                                                                            + np.mean(Z_obs, axis=1)))
        T_obs_probs = 1 / (1 + T_obs_scores)
        T_obs = np.random.binomial(1, T_obs_probs).squeeze()
        print("obs-number>0.95:", np.sum((T_obs_probs) > 0.95))
        print("obs-number<0.05:", np.sum((T_obs_probs) < 0.05))

        # generate T_exp
        T_exp_scores = np.exp((np.dot(X_exp, weight_x2t)).squeeze() / (np.mean(X_exp, axis=1)))
        T_exp_probs = 1 / (1 + T_exp_scores)
        T_exp = np.random.binomial(1, T_exp_probs).squeeze()
        print("obs-number>0.95:", np.sum((T_exp_probs) > 0.95))
        print("obs-number<0.05:", np.sum((T_exp_probs) < 0.05))


        T0_obs = np.zeros_like(T_obs)
        T1_obs = np.ones_like(T_obs)


        T0_exp = np.zeros_like(T_exp)
        T1_exp = np.ones_like(T_exp)


        noiseS_Y_obs = np.random.normal(0, 1, (Obsdata_size, 14)) * 0.7
        noiseS_Y_exp = np.random.normal(0, 1, (Expdata_size, 14)) * 0.7


        S_obs, Y_obs = self.Generate_Obs_S_Y(Obsdata_size, X_obs, T_obs, Z_obs, W_yj, beta_z2t, noiseS_Y_obs)

        # do(0) and do(1) calculate S_obs and Y_obs
        S1_obs, Y1_obs = self.Generate_Obs_S_Y(Obsdata_size, X_obs, T1_obs, Z_obs, W_yj, beta_z2t, noiseS_Y_obs)
        S0_obs, Y0_obs = self.Generate_Obs_S_Y(Obsdata_size, X_obs, T0_obs, Z_obs, W_yj, beta_z2t, noiseS_Y_obs)


        S_exp, Y_exp = self.Generate_Exp_S_Y(Expdata_size, X_exp, T_exp, Z_exp, W_yj, beta_z2t, noiseS_Y_exp)

        # do(0) and do(1) calculate S_exp and Y_exp
        S1_exp, Y1_exp = self.Generate_Exp_S_Y(Expdata_size, X_exp, T1_exp, Z_exp, W_yj, beta_z2t, noiseS_Y_exp)
        S0_exp, Y0_exp = self.Generate_Exp_S_Y(Expdata_size, X_exp, T0_exp, Z_exp, W_yj, beta_z2t, noiseS_Y_exp)

        Real_Obs_Ite_List = Y1_obs - Y0_obs
        Real_Obs_ATE = np.mean(Y1_obs - Y0_obs)

        Real_Exp_Ite_List = Y1_exp - Y0_exp
        Real_Exp_ATE = np.mean(Y1_exp - Y0_exp)


        return (torch.from_numpy(1.0 * U_obs), torch.from_numpy(Z_obs), torch.from_numpy(X_obs), torch.from_numpy(
            T_obs), torch.from_numpy(
            S_obs), torch.from_numpy(Y_obs),

                torch.from_numpy(1.0 * U_exp), torch.from_numpy(Z_exp), torch.from_numpy(X_exp), torch.from_numpy(
            T_exp), torch.from_numpy(
            S_exp), torch.from_numpy(Y_exp),

                Real_Obs_Ite_List, Real_Obs_ATE, Real_Exp_Ite_List, Real_Exp_ATE)

