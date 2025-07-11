import sys

import numpy
from sklearn.svm import SVR, SVC

sys.path.append('..')

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
import copy
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as reg_tree
from sklearn.ensemble import AdaBoostRegressor as ada_reg
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures as ply_fe


def poly_feature(X, degree=2):
    poly = ply_fe(degree=degree)
    return poly.fit_transform(X)


def get_best_for_data(X, Y):
    regs = [rfr(n_estimators=i) for i in [10, 20, 40, 60, 100, 150, 200]]
    regs += [reg_tree(max_depth=i) for i in [5, 10, 20, 30, 40, 50]]
    regs += [ada_reg(n_estimators=i) for i in [10, 20, 50, 70, 100, 150, 200]]
    regs += [gbr(n_estimators=i) for i in [50, 70, 100, 150, 200]]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    val_errs = []
    models = []
    for reg in regs:
        model = copy.deepcopy(reg)
        model.fit(x_train, y_train)
        val_errs.append(mse(y_test, model.predict(x_test)))
        models.append(copy.deepcopy(model))
    min_ind = val_errs.index(min(val_errs))
    # print(str(model)[:40], val_errs[min_ind])
    return copy.deepcopy(models[min_ind])


def e_x_estimator(x, w):
    """estimate P(W_i=1|X_i=x)"""
    log_reg = LogisticRegression().fit(x, w)
    return log_reg

def e_x_estimator_kernel_version(x, w, kernelType='rbf'):
    """estimate P(W_i=1|X_i=x)"""
    svr_k1 = SVC(kernel=kernelType,probability=True)
    log_reg = svr_k1.fit(x, w)
    return log_reg




def regression1D(x, y, type='linear', bias=True):
    # fit regression of E[Y|X]
    if type == 'linear':
        regression = LinearRegression(fit_intercept=bias).fit(X=x, y=y)
    elif type == 'kernelRidge':
        regression = KernelRidge().fit(X=x, y=y)
    elif type == 'randomForestRegressor':
        regression = RandomForestRegressor().fit(X=x, y=y)
    elif type == 'best':
        regression = get_best_for_data(x,y)
    else:
        raise Exception('undefined regression type')

    return regression


def naive_estimator(t, y):
    """estimate E[Y|T=1] - E[Y|T=0], would be biased if it's not RCTs"""
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)
    y1 = y[index_t1,]
    y0 = y[index_t0,]

    tau = np.mean(y1) - np.mean(y0)
    return tau


def ipw_estimator(x, t, y):
    """estimate ATE using ipw method"""
    propensity_socre_reg = e_x_estimator(x, t)
    propensity_socre = propensity_socre_reg.predict_proba(x)
    propensity_socre = propensity_socre[:, 1][:, None]  # prob of treatment=1

    ps1 = 1. / np.sum(t / propensity_socre)
    y1 = ps1 * np.sum(y * t / propensity_socre)
    ps0 = 1. / np.sum((1. - t) / (1. - propensity_socre))
    y0 = ps0 * np.sum(y * ((1. - t) / (1 - propensity_socre)))
    # print((1. - t).sum())
    # print(t.sum())

    tau = y1 - y0
    return tau


def s_learner_estimator(x, t, y,  type='linear'):
    """ estimate E(Y|X,T=1)-E(Y|X,T=0)
        s_learner: naive estimator using same regression function
    """
    # fit regression of E[Y|X]
    x_t = np.concatenate((x, t[:,None]), axis=1)
    if type == 'linear':
        regression = LinearRegression().fit(X=x_t, y=y)
    elif type == 'kernelRidge':
        regression = KernelRidge().fit(X=x_t, y=y)
    elif type == 'randomForestRegressor':
        regression = RandomForestRegressor().fit(X=x_t, y=y)
    elif type == 'best':
        regression = get_best_for_data(x_t,y)
    else:
        raise Exception('undefined regression type')
    x_t1 = np.concatenate((x, numpy.ones_like(t)[:,None]), axis=1)
    x_t0 = np.concatenate((x, numpy.zeros_like(t)[:,None]), axis=1)
    y1 = regression.predict(X=x_t1)
    y0 = regression.predict(X=x_t0)

    tau = y1 - y0
    return np.mean(tau)


def t_learner_estimator(x, t, y, type='linear'):
    """ estimate E(Y|X,T=1)-E(Y|X,T=0)
        t_learner: naive estimator using different regression function
    """
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)
    x_t1 = x[index_t1]
    x_t0 = x[index_t0]

    if type == 'linear':
        regression_1 = LinearRegression().fit(X=x_t1, y=y[index_t1,])
        regression_0 = LinearRegression().fit(X=x_t0, y=y[index_t0,])
    elif type == 'kernelRidge':
        regression_1 = KernelRidge().fit(X=x_t1, y=y[index_t1,])
        regression_0 = KernelRidge().fit(X=x_t0, y=y[index_t0,])
    elif type == 'randomForestRegressor':
        regression_1 = RandomForestRegressor().fit(X=x_t1, y=y[index_t1,])
        regression_0 = RandomForestRegressor().fit(X=x_t0, y=y[index_t0,])
    elif type == 'best':
        regression_1 = get_best_for_data(x_t1, y[index_t1,])
        regression_0 = get_best_for_data(x_t0, y[index_t0,])
    else:
        raise Exception('undefined regression type')

    y1 = regression_1.predict(X=x)
    y0 = regression_0.predict(X=x)

    tau = np.mean(y1 - y0)
    return tau


def x_learner_estimator():
    pass


def double_robust_estimator(x, t, y):
    pass


def tmle_estimator(x, t, y):
    pass






