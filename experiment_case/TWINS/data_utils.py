import numpy as np
import pandas as pd
import random


# ------------------------------- IHDP --------------------------------
def load_ihdp_npz(idx):
    tr_data = np.load('data/ihdp_npci_1-1000.train.npz')
    te_data = np.load('data/ihdp_npci_1-1000.test.npz')

    ## concatenate original training and test data
    A = np.concatenate((tr_data['t'][:, idx], te_data['t'][:, idx]))
    X = np.concatenate((tr_data['x'][:, :, idx], te_data['x'][:, :, idx]))
    Y = np.concatenate((tr_data['yf'][:, idx], te_data['yf'][:, idx]))
    Y0 = np.concatenate((tr_data['mu0'][:, idx], te_data['mu0'][:, idx]))
    Y1 = np.concatenate((tr_data['mu1'][:, idx], te_data['mu1'][:, idx]))

    ## preprocess X (scaling + PCA)
    X_neighbor = pipe.fit_transform(X)

    ## add some noise to outcome Y
    noise1 = self.args.noise * np.random.normal(0, 1, size=len(X))
    noise0 = self.args.noise * np.random.normal(0, 1, size=len(X))
    Y = (Y1 + noise1) * A + (Y0 + noise0) * (1 - A)

    ind = np.arange(len(X))
    tr_ind = ind[:470]
    val_ind = ind[470:670]
    te_ind = ind[670:747]

    return tr_ind, val_ind, te_ind

def load_ihdp(dir="./dataset/IHDP/", idx=1, train_ratio=0.2):
    random.seed(0)
    np.random.seed(0)

    data = pd.read_csv(dir + "ihdp_npci_{}.csv".format(idx))
    idx = list(range(len(data)))
    random.shuffle(idx)  # 将index列表打乱
    train_len = int(len(data) * train_ratio)
    data_train = data.loc[idx[:train_len]]
    data_test = data.loc[idx[train_len:]]

    x_cols = ["x"+str(num) for num in range(1, 26)]

    # get train data
    X_train = data_train.loc[:, x_cols].values
    t_train, y_train = data_train["t"].values, data_train["y"].values
    mu0_train, mu1_train = data_train["mu0"].values, data_train["mu1"].values
    att_train = (mu1_train[t_train==1] - mu0_train[t_train==1]).mean()

    # get test data
    X_test = data_test.loc[:, x_cols].values
    t_test, y_test = data_test["t"].values, data_test["y"].values
    mu0_test, mu1_test = data_test["mu0"].values, data_test["mu1"].values
    att_test = (mu1_test[t_test == 1] - mu0_test[t_test == 1]).mean()

    return X_train, t_train, y_train, att_train, X_test, t_test, y_test, att_test


# ------------------------------- Twin --------------------------------
def load_twins(dir="./dataset/twins/", train_ratio=0.5, seed=0):
    data = pd.read_csv(dir + "twins_data_cleaned.csv")

    # define the size of training data
    train_len = int(len(data) * train_ratio)

    # split the data
    random.seed(seed)
    np.random.seed(seed)
    idx = list(range(len(data)))
    random.shuffle(idx)
    data_train = data.loc[idx[:train_len]]
    data_test = data.loc[idx[train_len:]]

    # get train data
    X_train = data_train.iloc[:, :30].values
    t_train, y_train = data_train["t"].values, data_train["y"].values
    mu0_train, mu1_train = data_train["mu0"].values, data_train["mu1"].values
    att_train = (mu1_train[t_train==1] - mu0_train[t_train==1]).mean()

    # get test data
    X_test = data_test.iloc[:, :30].values
    t_test, y_test = data_test["t"].values, data_test["y"].values
    mu0_test, mu1_test = data_test["mu0"].values, data_test["mu1"].values
    att_test = (mu1_test[t_test == 1] - mu0_test[t_test == 1]).mean()

    return X_train, t_train, y_train, att_train, X_test, t_test, y_test, att_test
