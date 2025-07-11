import numpy as np
# from scipy.special import expit as sigmoid
# #import igraph as ig
# import random
# import networkx as nx
# import torch
import matplotlib.pyplot as plt
# import math
# import os
import pandas as pd
#from brokenaxes import brokenaxes

def draw_line_pdf(num_line, data, index_X, xlabel, ylabel,data_name=None, ylim=(-0.05, 1.09), loc=(0.7,0.7),
                  file_name = None, is_show=False, is_save=False, title=None, ylable_fontsize="13"
                  ,draw_hline=False):
    """
    df = pd.DataFrame(np.array([[0.2       , 0.4       , 0.9       , 0.94      , 0.94      ,
        0.98      , 1.        ],
       [0.33      , 0.39      , 0.82      , 0.92      , 0.92      ,
        0.99      , 1.        ],
       [0.32666667, 0.39333333, 0.78666667, 0.86666667, 0.85333333,
        0.94666667, 0.99333333],
       [0.325     , 0.465     , 0.785     , 0.855     , 0.83      ,
        0.965     , 0.995     ]]),index=['0.5', '1.0', '1.5', '2.0'],columns=['PCMCI', 'NHPC', 'MLE_SGL', 'ADM4', 'TTHP_NT', 'TTHP_S', 'TTHP'])

    :param num_line:
    :param data:
    :param index_X:
    :param data_name:
    :param xlabel:
    :param ylabel:
    :param file_name:
    :return:
    """
    error_config = {'ecolor': '0.3', 'capsize': 2}
    # marker = ['-x', '-D', '-*', '-s', '-v', '-o', '-^']
    marker = ['-o','-*']
    marker = marker[:num_line]
    if data_name is None:
        df = pd.DataFrame(data, index=index_X)
    else:
        df = pd.DataFrame(data, index=index_X, columns=data_name,)
    ax = df.plot(kind='line', style=marker,figsize=(5,2.3), ylim=ylim, rot=0) # 这里面的line可以改成bar之类的
    ax.grid( linestyle="dotted") # 设置背景线\
    if data_name is None:
        ax.legend().remove()
    if data_name is not None:
        ax.legend(fontsize=9, loc=loc) # 设置图例位置
    ax.set_xlabel(xlabel, fontsize='13')
    ax.set_ylabel(ylabel, fontsize=ylable_fontsize)
    if title is not None:
        plt.title(label=title, fontsize='17', loc='center')
    if draw_hline is True:
        plt.hlines(0,index_X[0],index_X[-1], colors="red")
    plt.xticks(index_X, fontsize=8)
    plt.axhline(y=0, c='black', ls=':')

    if is_save and file_name is not None:
        plt.savefig("graph/" + file_name, format='pdf', bbox_inches='tight')
        df.to_excel("graph/" + file_name + '.xlsx')
    if is_show:
        plt.show()


# 示例数据
data = np.array([[0.526, 0.933, 1.096, 0.334, 0.531, 0.576, 0.143],
                 [0.937, 0.975, 1.203, 0.609, 1.115, 0.999, 0.183],
                 [1.347, 2.603, 1.525, 0.875, 1.732, 1.604, 0.219],
                 [2.779, 3.757, 2.683,2.589 , 4.280,4.212 ,0.321 ],
                 [11.320, 28.672, 8.648, 18.648,18.991, 18.627, 2.045]
                 ])
index_X = ['1', '1.5', '2', '3','5']
data_name = ['S-leaner', 'T-leaner', 'equi_con', 'LTEE', 'TEDVAE', 'CEVAE', 'Our Method(ICEVAE)']

# 调用函数绘制图表
draw_line_pdf(num_line=7,
              data=data,
              index_X=index_X,
              xlabel='X Label',
              ylabel='Y Label',
              data_name=data_name,
              ylim=(0, 13),
              loc=(0.7, 0.7),
              file_name='example.pdf',
              is_show=True,
              is_save=False,
              title='Example Title',
              ylable_fontsize='13',
              draw_hline=True)



data_ate = np.array([[0.0771064415516062,0.24957221227097529,5.303477208338691,12.330020076231794,0.09852081176551195,0.08229008643504397,0.13271369539866287,0.13223546146169105,0.11872421547925563,0.08464842257236817],
                 [0.1778125466544045,0.27015116328303374,6.973679185028736,9.591962336758495,0.06689581307952748,0.4375311851360501,0.36472515435199143,0.39521664638921994,0.3475622419883047, 0.10421285180516753],
                 [0.3138170005974198,1.4680682865907475,7.398088849648751,11.215543557423038,0.1961222646180949,0.4842872552758597 ,0.5145547661574101,0.7272423486651031,0.6599467287025965,0.1150685718544963],
                 [0.756133643194925,0.8907365985879888,8.496975357867921,9.939110852558823,0.2088115757532807,1.1133056154440633 ,1.7584635847225794,2.3787462097703003,2.325020596072922,0.206830502695168],
                 [7.002660054578655,13.810292869752995,14.994804065358574,40.59354640475037,4.332453179152585,17.398567690494154,13.349527924643265,13.411182560979956,12.744992851102225,1.516283312628244]
                 ])
index_X = ['1', '1.5', '2', '3','5']
data_name_ate = ['S-leaner', 'T-leaner','imp_MLP','weighting', 'equi_con','ifbase', 'LTEE', 'TEDVAE', 'CEVAE', 'Our Method(ICEVAE)']

# 调用函数绘制图表
draw_line_pdf(num_line=10,
              data=data_ate,
              index_X=index_X,
              xlabel='X Label',
              ylabel='Y Label',
              data_name=data_name_ate,
              ylim=(0, 12),
              loc=(0.7, 0.7),
              file_name='example.pdf',
              is_show=True,
              is_save=False,
              title='Example Title',
              ylable_fontsize='13',
              draw_hline=True)