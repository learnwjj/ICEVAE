import sys
from other_model.ate_estimator import e_x_estimator, regression1D, ipw_estimator, t_learner_estimator, s_learner_estimator
import numpy
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor


# definition
# k(w,x,y,O) = E[Y|w,x,s,G=O]
# r = pr(G=O|x)
# RCT G=1, OBS G=0

def imputationApproach(X, S, Y, T, G, regressionType='linear'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    k1 = regression1D(np.concatenate((XO[TO[:,0]==1,:], SO[TO[:,0]==1,:]), 1), YO[TO[:,0]==1,:], type=regressionType)
    k0 = regression1D(np.concatenate((XO[TO[:,0]==0,:], SO[TO[:,0]==0,:]), 1), YO[TO[:,0]==0,:], type=regressionType)
    r = e_x_estimator(X, G)  # E[G|X]

    # print(k1.coef_, k1.intercept_)
    # print(k0.coef_, k0.intercept_)

    ratio = r.predict_proba(XE)[:, 0][:, None] / r.predict_proba(XE)[:, 1][:, None]  # p(G=0|X)/p(G=E|X)

    tau = np.sum(TE * k1.predict(np.concatenate((XE, SE), 1)) * ratio) / np.sum(TE * ratio) \
          - np.sum((1 - TE) * k0.predict(np.concatenate((XE, SE), 1)) * ratio) / np.sum(
        (1 - TE) * ratio)
    return tau


def weightingApproach(X, S, Y, T, G, kernelType='gaussian'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    from sklearn.neighbors import KernelDensity
    kdeO1 = KernelDensity(kernel=kernelType, bandwidth=0.2).fit(np.concatenate((TO, SO, XO), 1))
    kdeO2 = KernelDensity(kernel=kernelType, bandwidth=0.2).fit(XO)
    kdeE1 = KernelDensity(kernel=kernelType, bandwidth=0.2).fit(np.concatenate((TE, SE, XE), 1))
    kdeE2 = KernelDensity(kernel=kernelType, bandwidth=0.2).fit(XE)

    # score_samples return log-likelihood
    weight_lambda = kdeE1.score_samples(np.concatenate((TO, SO, XO), 1)) - kdeE2.score_samples(XO) - \
                    kdeO1.score_samples(np.concatenate((TO, SO, XO), 1)) + kdeO2.score_samples(XO)
    weight_lambda = np.exp(weight_lambda)

    ex = e_x_estimator(np.concatenate((X, G[:, None]), 1), T)
    ex_proba = ex.predict_proba(np.concatenate((XO, np.ones_like(TO)), 1))[:, 1][:, None]  # prob of treatment=1

    # ex = e_x_estimator(XE, TE)
    # ex_proba = ex.predict_proba(XO)[:, 1][:, None]

    # tau = np.sum(YO * TO * weight_lambda / ex_proba) / np.sum((1-TO) * weight_lambda / ex_proba) -\
    #     np.sum(YO * (1-TO) * weight_lambda / (1-ex_proba)) / np.sum(TO * weight_lambda / (1-ex_proba))

    tau = np.sum(YO * TO * weight_lambda[:,None] / ex_proba) / np.sum(TO * weight_lambda[:,None] / ex_proba) - \
          np.sum(YO * (1 - TO) * weight_lambda[:,None] / (1 - ex_proba)) / np.sum((1 - TO) * weight_lambda[:,None] / (1 - ex_proba))

    return tau


def weightingApproach_(X, S, Y, T, G, kernelType='gaussian'):
    # different: using stat lib
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    from scipy import stats
    kdeO1 = stats.gaussian_kde(dataset=np.concatenate((TO, SO, XO), 1).T)
    kdeO2 = stats.gaussian_kde(dataset=XO.T)
    kdeE1 = stats.gaussian_kde(dataset=np.concatenate((TE, SE, XE), 1).T)
    kdeE2 = stats.gaussian_kde(dataset=XE.T)

    # score_samples return log-likelihood
    weight_lambda = kdeE1.pdf(np.concatenate((TO, SO, XO), 1).T) - kdeE2.pdf(XO.T) - \
                    kdeO1.pdf(np.concatenate((TO, SO, XO), 1).T) + kdeO2.pdf(XO.T)
    # weight_lambda = np.exp(weight_lambda)

    ex = e_x_estimator(np.concatenate((X, G[:, None]), 1), T)
    ex_proba = ex.predict_proba(np.concatenate((XO, np.ones_like(TO)), 1))[:, 1][:, None]  # prob of treatment=1

    tau = np.sum(YO * TO * weight_lambda[:,None] / ex_proba) / np.sum(TO * weight_lambda[:,None] / ex_proba) - \
          np.sum(YO * (1 - TO) * weight_lambda[:,None] / (1 - ex_proba)) / np.sum((1 - TO) * weight_lambda[:,None] / (1 - ex_proba))

    return tau


def con_cdf(kde_join, kde_con, S,T,X, sample_gap=1e-1):
    '''
    sample uniformly s, output cdf of P(S|T,X)
    :param kde_join: p(s,t,x)
    :param kde_con:  p(t,x)
    :param S:
    :param T:
    :param X:
    :return:
    '''
    # for each S, uniformly sample and calculate integrate using mcmc
    Smin, Smax = -100, np.max(S)
    evenly_array = np.linspace(start=Smin, stop=Smax, num=int((Smax-Smin)/sample_gap), endpoint=True)
    cdf = list()
    for i,j,k in zip(S,T,X):
        # print(i,j,k)
        # print(Smin)

        index = int((i-Smin)/sample_gap)

        # print(index)
        # print(length)

        evenly_array_i = evenly_array[:index]
        tall = 0
        con_cdf_log_value_i = kde_con.score_samples(np.expand_dims(np.concatenate((j, k), 0),0))

        for tall_i in evenly_array_i:
            tall += np.exp(kde_join.score_samples(np.expand_dims(np.concatenate(([tall_i], j, k), 0),0)) - con_cdf_log_value_i)

        # tall = tall - np.exp((kde_join.score_samples(np.expand_dims(np.concatenate((i, j, k), 0),0))- con_cdf_log_value_i))/2 -\
        #             np.exp((kde_join.score_samples(np.expand_dims(np.concatenate(([Smin], j, k), 0),0)) - con_cdf_log_value_i))/2

        cdf_i = tall * sample_gap
        cdf.append(cdf_i)
    cdf = np.array(cdf)
    return cdf

def controlFunctionApproach(X, S, Y, T, G, regressionType='linear', kernelType='gaussian'):
    # devide into G=O and G=E
    XO, SO, YO, TO = X[G == 0], S[G == 0][:, None], Y[G == 0][:, None], T[G == 0][:, None]
    XE, SE, TE = X[G == 1], S[G == 1][:, None], T[G == 1][:, None]

    S, T, G = S[:, None], T[:, None], G[:, None]

    from sklearn.neighbors import KernelDensity
    kde_join = KernelDensity(kernel=kernelType, bandwidth=0.2).fit(np.concatenate((SE, TE, XE), 1))
    kde_condition = KernelDensity(kernel=kernelType, bandwidth=0.2).fit(np.concatenate((TE, XE), 1))

    etaO = con_cdf(kde_join, kde_condition, SO, TO, XO)
    etaE = con_cdf(kde_join, kde_condition, SE, TE, XE)

    # print(np.min(etaE), np.max(etaE))
    # print(np.min(etaO), np.max(etaO))

    gamma = regression1D(np.concatenate((XO, etaO, TO), 1), YO, type=regressionType)

    tau = np.mean(TE * gamma.predict(np.concatenate((XE, etaE, TE), 1))) - \
          np.mean((1 - TE) * gamma.predict(np.concatenate((XE, etaE, TE), 1)))
    return tau

def dgp():
    Osize, Esize = 500, 500
    # XO XE can be slightly different, but UO and UE should be totally same, i.e., X->G but not U->G
    XO, UO = np.random.multivariate_normal(mean=[2, 1], cov=[[1, 0], [0, 1]], size=Osize), \
        np.random.multivariate_normal(mean=[-1, 0], cov=[[1, 0], [0, 1]], size=Osize)
    XE, UE = np.random.multivariate_normal(mean=[1, 0], cov=[[1, 0], [0, 1]], size=Esize), \
        np.random.multivariate_normal(mean=[-1, 0], cov=[[1, 0], [0, 1]], size=Esize)

    TO = np.random.binomial(n=1, size=Osize, p=(1 / (1 + np.exp(-np.mean(np.concatenate((XO, 3 * UO), 1), axis=1)))))
    TE = np.random.binomial(n=1, size=Esize, p=0.5)

    # rule out extreme data
    # indexO = np.logical_and(((1 / (1 + np.exp(-np.mean(np.concatenate((XO, UO), 1), axis=1)))) < 0.95),
    #                         ((1 / (1 + np.exp(-np.mean(np.concatenate((XO, UO), 1), axis=1)))) > 0.05))
    # indexE = np.logical_and(((1 / (1 + np.exp(-np.mean(XE, axis=1)))) < 0.95),
    #                         ((1 / (1 + np.exp(-np.mean(XE, axis=1)))) > 0.05))
    # XO, UO = XO[indexO, :], UO[indexO, :]
    # XE, UE = XE[indexE, :], UE[indexE, :]
    # TO, TE = TO[indexO], TE[indexE]
    # Osize, Esize = TO.shape[0], TE.shape[0]

    print('sizeO,E=' + str(Osize) + str(',') + str(Esize))

    noiseSO, noiseSE = np.random.randn(Osize) * 0.7, np.random.randn(Esize) * 0.7
    SO_treated = 3 + np.mean(XO, axis=1) + 2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + noiseSO
    SO_control = 2 * np.mean(XO, axis=1) + np.mean(UO, axis=1) + noiseSO
    SE_treated = 3 + np.mean(XE, axis=1) + 2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + noiseSE
    SE_control = 2 * np.mean(XE, axis=1) + np.mean(UE, axis=1) + noiseSE

    noiseO, noiseE = np.random.randn(Osize) * 0.7, np.random.randn(Esize) * 0.7
    YO_treated = 4 + np.mean(XO, axis=1) + np.mean(XO, axis=1) + SO_treated + noiseO
    YO_control = np.mean(XO, axis=1) + SO_control + noiseO
    YE_treated = 4 + np.mean(XE, axis=1) + 5 * np.mean(UE, axis=1) + np.mean(XE, axis=1) + SE_treated + noiseE
    YE_control = np.mean(XE, axis=1) + SE_control + noiseE

    SE, SO = np.where(TE==1, SE_treated, SE_control), np.where(TO==1, SO_treated, SO_control)
    YE, YO = np.where(TE==1, YE_treated, YE_control), np.where(TO==1, YO_treated, YO_control)

    ite_O, ite_E = YO_treated-YO_control, YE_treated-YE_control

    X, T, S, Y, G = np.concatenate((XO, XE), 0), np.concatenate((TO, TE), 0), np.concatenate((SO, SE), 0), \
        np.concatenate((YO, YE), 0), \
        np.concatenate((np.zeros_like(YO), np.ones_like(YE)), 0).squeeze()

    return X, S, Y, T, G,\
            XO, TO, YO,\
            XE, TE, YE,\
            ite_O, ite_E

def test():
    X, S, Y, T, G, \
        XO, TO, YO, \
        XE, TE, YE,\
        ite_O, ite_E = dgp()


    print('grt_O: ' + str(np.mean(ite_O)))
    print('grt_E: ' + str(np.mean(ite_E)))

    print('ipw using obs data: '+str(ipw_estimator(XO, TO, YO)))
    print('ipw using exp data: '+str(ipw_estimator(XE, TE, YE)))
    print('s_learner using exp data: '+str(s_learner_estimator(XE, TE, YE)))
    print('imputationApproach: '+str(imputationApproach(X, S, Y, T, G, regressionType='linear')))
    print('weightingApproach: '+str(weightingApproach(X, S, Y, T, G)))
    print('weightingApproach_: '+str(weightingApproach_(X, S, Y, T, G)))
    #print('controlFunctionApproach: '+str(controlFunctionApproach(X, S, Y, T, G, regressionType='kernelRidge')))


test()
