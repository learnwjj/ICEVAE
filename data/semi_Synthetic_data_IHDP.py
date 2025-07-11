
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

    # this method generate obs short and long term outcome
    def Generate_Obs_S_Y(self, Obsdata_size_input, x, t, z, W_yj, beta_z2t,
                         noiseS_Y):
        Obsdata_size = Obsdata_size_input

        part1 = np.sum(W_yj[5:9] * x[:, :4] , axis=1)

        part2 = 0.25*np.sum(W_yj[10:13] * x[:, 4:7], axis=1)

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
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + 0.25 * np.mean(S[:, :i],
                                                                                                          axis=1) + noiseS_Y[
                                                                                                                    :,
                                                                                                                    i]

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

    # this method generate exp short and long term outcome
    def Generate_Exp_S_Y(self, Obsdata_size_input, x, t, z, W_yj, beta_z2t, noiseS_Y):
        Obsdata_size = Obsdata_size_input

        part1 = np.sum(W_yj[5:9] * x[:, :4], axis=1)

        part2 = 0.25 * np.sum(W_yj[10:13] * x[:, 4:7], axis=1)

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
                S_i = part1 + ((i + 1) + 5) * t * part2 + beta_z2t / 5 * (i + 1) * part3 + 0.25 * np.mean(S[:, :i],
                                                                                                          axis=1) + noiseS_Y[
                                                                                                                    :,
                                                                                                                    i]
            # 更新历史
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

    # this method return exp and obs data
    def GenerateDataset_STDmethod(self, dir="../data/IHDP/csv/",obs_ratio=0.6, seed=0):

        random.seed(seed)
        np.random.seed(seed)
        data1 = pd.read_csv(dir + "ihdp_npci_{}.csv".format(2*seed))
        data2 = pd.read_csv(dir + "ihdp_npci_{}.csv".format(2*seed+1))
        data=   pd.concat([data1, data2], ignore_index=True)

        idx = list(range(len(data)))
        random.shuffle(idx)
        obs_len = int(len(data) * obs_ratio)
        exp_len = int(len(data)-obs_len)
        data_obs = data.loc[idx[:obs_len]]
        data_exp = data.loc[idx[obs_len:]]

        U_pre_obs = data_obs[["x10","x11","x12","x19", "x20", "x21", "x22", "x23","x24","x25"]].values
        binary_values_U_pre_obs = U_pre_obs[:, 3:]
        modifiers_U_pre_obs = U_pre_obs[:, :3]
        U_obs_temp = np.array([int("".join(map(str, row)), 2) for row in binary_values_U_pre_obs])
        U_obs = np.zeros_like(U_obs_temp, dtype=float)
        for i in range(len(U_obs_temp)):
            modified_value = U_obs_temp[i]
            if modifiers_U_pre_obs[i, 0] == 1:
                modified_value *= 3
            if modifiers_U_pre_obs[i, 1] == 1:
                modified_value *= 2
            if modifiers_U_pre_obs[i, 2] == 1:
                modified_value *= 1
            U_obs[i] = modified_value
        U_obs=U_obs / 10
        Z_obs = data_obs[["x13", "x15", "x16", "x17", "x18"]].values
        X_obs = data_obs[["x1", "x2", "x3", "x4", "x5", "x7", "x8"]].values



        U_pre_exp = data_exp[["x10", "x11", "x12", "x19", "x20", "x21", "x22", "x23", "x24","x25"]].values
        binary_values_U_pre_exp = U_pre_exp[:, 3:]
        modifiers_U_pre_exp = U_pre_exp[:, :3]
        U_exp_temp = np.array([int("".join(map(str, row)), 2) for row in binary_values_U_pre_exp])
        U_exp = np.zeros_like(U_exp_temp, dtype=float)
        for i in range(len(U_exp_temp)):
            modified_value = U_exp_temp[i]
            if modifiers_U_pre_exp[i, 0] == 1:
                modified_value *= 3
            if modifiers_U_pre_exp[i, 1] == 1:
                modified_value *= 2
            if modifiers_U_pre_exp[i, 2] == 1:
                modified_value *= 1
            U_exp[i] = modified_value
        U_exp=U_exp/10
        Z_exp = data_exp[["x13", "x15", "x16", "x17", "x18"]].values
        X_exp = data_exp[["x1", "x2", "x3", "x4", "x5", "x7", "x8"]].values

        Obsdata_size=obs_len
        Expdata_size=exp_len



        W_tj = np.random.uniform(0, 0.5, 10)
        W_yj = np.random.uniform(0.5, 1, 15)
        beta_z2t = 0.25


        T_values = [0, 1]
        weight_x2t = np.random.uniform(low=-0.5, high=0.5, size=(7, 1))
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


        S1_obs, Y1_obs = self.Generate_Obs_S_Y(Obsdata_size, X_obs, T1_obs, Z_obs, W_yj, beta_z2t, noiseS_Y_obs)
        S0_obs, Y0_obs = self.Generate_Obs_S_Y(Obsdata_size, X_obs, T0_obs, Z_obs, W_yj, beta_z2t, noiseS_Y_obs)


        S_exp, Y_exp = self.Generate_Exp_S_Y(Expdata_size, X_exp, T_exp, Z_exp, W_yj, beta_z2t, noiseS_Y_exp)


        S1_exp, Y1_exp = self.Generate_Exp_S_Y(Expdata_size, X_exp, T1_exp, Z_exp, W_yj, beta_z2t, noiseS_Y_exp)
        S0_exp, Y0_exp = self.Generate_Exp_S_Y(Expdata_size, X_exp, T0_exp, Z_exp, W_yj, beta_z2t, noiseS_Y_exp)

        Real_Obs_Ite_List = Y1_obs - Y0_obs
        Real_Obs_ATE = np.mean(Y1_obs - Y0_obs)

        Real_Exp_Ite_List = Y1_exp - Y0_exp
        Real_Exp_ATE = np.mean(Y1_exp - Y0_exp)


        return (torch.from_numpy(1.0 * U_obs[:,None]), torch.from_numpy(1.0 *Z_obs), torch.from_numpy(1.0 *X_obs), torch.from_numpy(
            T_obs), torch.from_numpy(
            S_obs), torch.from_numpy(Y_obs),

                torch.from_numpy(1.0 * U_exp[:,None]), torch.from_numpy(1.0 *Z_exp), torch.from_numpy(1.0 *X_exp), torch.from_numpy(
            T_exp), torch.from_numpy(
            S_exp), torch.from_numpy(Y_exp),

                Real_Obs_Ite_List, Real_Obs_ATE, Real_Exp_Ite_List, Real_Exp_ATE)

