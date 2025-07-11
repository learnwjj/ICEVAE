from numbers import Number
'''
我们写的不使用pyro的iCevae版本
'''
import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch.distributions import bernoulli, normal
from dataset import DGP


class MLP(nn.Module):
    def __init__(self, in_dim,out_dim=None,hid_dim=None,device='cpu', activation="gelu", layer_norm=True, activation_exit=True,layer_nums=2,add_sigmiod=False):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        elif activation == "leaky_relu":
            a_f = nn.LeakyReLU()
        else:
            a_f = nn.Identity()
        if out_dim is None:
            out_dim = in_dim
        dropout_net = nn.Dropout(0.05)
        if layer_nums == 1:
            net = [nn.Linear(in_dim, out_dim)]
        else:

            # net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
            #     nn.Linear(in_dim, hid_dim), a_f]
            #########################################
            net = []
            net.append(nn.Linear(in_dim, hid_dim))
            if activation_exit:
                net.append(a_f)
            if layer_norm:
                net.append(nn.LayerNorm(hid_dim))
            ####################################

            for i in range(layer_norm - 2):
                net.append(nn.Linear(in_dim, hid_dim))
                net.append(a_f)
                net.append(dropout_net)
            net.append(nn.Linear(hid_dim, out_dim))

        if add_sigmiod:
           net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# class MLP(torch.nn.Module):#对nn.module进行继承，也就是基于pytorch的网络的模板类进行继承，这个类描述了叫MLP的神经网络结构
#
#     def __init__(self, input_dim, output_dim, hidden_dim, slope=.1, device='cpu',activation="gelu"):
#          # __init__ 方法是类的构造函数，用于初始化类的实例。它接受一些参数，包括 input_dim（输入维度）、output_dim（输出维度）、hidden_dim（隐藏层维度或列表）、n_layers（隐藏层数）、activation（激活函数类型或列表）、slope（斜率）和 device（设备类型，默认为CPU）。
#         super().__init__()  # 初始化网络参数，下面四行是另外声明的变量
#         self.input_dim = input_dim  # 输入
#         self.output_dim = output_dim  # 输出
#         self.hidden_dim = hidden_dim #
#         self.device = device  # 设备类型如cpu和GPU
#         self.fc1=torch.nn.Linear(self.input_dim,self.hidden_dim)
#         self.fc2=torch.nn.Linear(self.hidden_dim,self.output_dim)
#         #
#         # torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
#         # torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
#         # torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
#         # torch.nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
#
#
#     def forward(self,x):
#         h=x
#         #dout1 = self.fc1(h)
#         dout1 = torch.nn.functional.leaky_relu(self.fc1(h))
#         return self.fc2(dout1)

class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass

class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        #torch.one()用1填充的张量，np.pi就是圆周率，to表示移送到指定pytorch设备上，这里填充的是一行的张量所以是（1）
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        #表示了一个标准正态分布对象
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):#从正态分布中采样，这里就是重参数的搞法，mu是均值，v是方差，采样的eps是epsilon
        eps = self._dist.sample(mu.size()).squeeze()#从标准正态分布中采样出一个和mu相同size的样本eps
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)#返回的就是u+eps*sqrt(v),eps~N(0,1)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)#改变张量的形状变为param_shape的形状
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))#正态分布的log后的函数表示，输入的x可能不是一个一维数据
        if reduce:
            return lpdf.sum(dim=-1)#返回求出的logP（）的总和
        else:
            return lpdf#返回logP（）的列表

class ICEVAE(nn.Module):
    DGP=DGP()


    def __init__(self, U_dim, Z_dim, X_dim, W_dim, S_dim, Y_dim,hidden_dim=264, prior=None, decoder=None, encoder=None, activation="gelu",  device='cpu',layer_norm=True, activation_exit=True,layer_nums=2,bin_using_sigmoid=False):
        super().__init__()

        self.U_dim = U_dim
        self.Z_dim = Z_dim
        self.X_dim = X_dim
        self.W_dim = W_dim
        self.S_dim = S_dim
        self.Y_dim = Y_dim
        self.device=device


        #######这段是判断预设的分布类型，没有规定默认高斯分布
        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder
        #######################################################

        #######先验分布P(Z|U),默认N(0,1)
        # self.prior_mean=torch.zeros(1).to(device)
        self.prior_mean=MLP(U_dim,Z_dim,hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.logl=MLP(U_dim,Z_dim,hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)#hidden层的宽度就和输入的宽度一致



        #######encoder params
        ##Z=m(X,S,W,U)
        self.z_wxsu_common_mean_en = MLP(X_dim+S_dim+U_dim, hidden_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.z_wxsu_common_logv_en = MLP(X_dim + S_dim + U_dim, hidden_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.z_w0xsu_mean_en = MLP(hidden_dim, Z_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.z_w0xsu_logv_en = MLP(hidden_dim, Z_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.z_w1xsu_mean_en = MLP(hidden_dim, Z_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.z_w1xsu_logv_en = MLP(hidden_dim, Z_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        #encoder的S(x,w)也需要估计，这里设置一个估计S()的网络
        self.s_wx_common_mean_en= MLP(X_dim, hidden_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.s_wx_common_logv_en = MLP(X_dim, hidden_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.s_w0x_mean_en = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=0xz for s's mean

        self.s_w0x_logv_en = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # #condtion w=0xz for s's var

        self.s_w1x_mean_en = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=1xz for s's mean

        self.s_w1x_logv_en = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # #condtion w=1xz for s's var




        ######decoder params

        #因为X由Z和U指向，所以需要两个condi
        self.x_zu_mean_de=MLP(Z_dim+U_dim,X_dim,hidden_dim,device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.x_zu_logv_de=MLP(Z_dim+U_dim,X_dim,hidden_dim,device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        if bin_using_sigmoid:
            self.w_xz_p_de = MLP(X_dim+Z_dim, W_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation,add_sigmiod=True)#w是binary
        else:
            self.w_xz_p_de = MLP(X_dim + Z_dim, W_dim, hidden_dim, device=device, layer_norm=layer_norm,activation_exit=activation_exit, layer_nums=layer_nums, activation=activation,add_sigmiod=False)  # w是binary


        self.s_wxz_common_mean_de= MLP(X_dim + Z_dim, hidden_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.s_wxz_common_logv_de = MLP(X_dim + Z_dim, hidden_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.s_w0xz_mean_de = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=0xz for s's mean

        self.s_w0xz_logv_de = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # #condtion w=0xz for s's var

        self.s_w1xz_mean_de = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=1xz for s's mean

        self.s_w1xz_logv_de = MLP(hidden_dim, S_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # #condtion w=1xz for s's var



        self.y_wzsx_common_mean_de=MLP(Z_dim + S_dim + X_dim, hidden_dim, hidden_dim,device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.y_wzsx_common_logv_de=MLP(Z_dim + S_dim + X_dim, hidden_dim, hidden_dim,device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)

        self.y_w0zsx_mean_de = MLP(hidden_dim, Y_dim, hidden_dim,device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=0zsx for y's mean

        self.y_w0zsx_logv_de = MLP(hidden_dim, Y_dim, hidden_dim ,device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=0zsx for y's var

        self.y_w1zsx_mean_de = MLP(hidden_dim, Y_dim, hidden_dim,device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=0zsx for y's mean

        self.y_w1zsx_logv_de = MLP(hidden_dim, Y_dim, hidden_dim, device=device,layer_norm=layer_norm, activation_exit=activation_exit,layer_nums=layer_nums,activation=activation)  # condtion w=0zsx for y's var

        self.bce_loss=nn.BCELoss(reduction='mean')

    def train_ens_parameters_exp(self):
        return list(self.s_wx_common_mean_en.parameters())+list(self.s_wx_common_logv_en.parameters())+\
         list(self.s_w0x_mean_en.parameters())+list(self.s_w0x_logv_en.parameters())+\
         list(self.s_w1x_mean_en.parameters())+list(self.s_w1x_logv_en.parameters())


    def train_parameters_obs(self):
        return (list(self.z_wxsu_common_mean_en.parameters())+list(self.z_wxsu_common_mean_en.parameters())+
                list(self.z_w0xsu_mean_en.parameters())+list(self.z_w0xsu_logv_en.parameters())
                +list(self.z_w1xsu_mean_en.parameters())+list(self.z_w1xsu_logv_en.parameters())+
        list(self.x_zu_mean_de.parameters())+list(self.x_zu_logv_de.parameters())+
        list(self.w_xz_p_de.parameters())+
        list(self.s_wxz_common_mean_de.parameters())+ list(self.s_wxz_common_logv_de.parameters())+
        list(self.s_w0xz_mean_de.parameters())+list(self.s_w0xz_logv_de.parameters())
        +list(self.s_w1xz_mean_de.parameters())+list(self.s_w1xz_logv_de.parameters())+
        list(self.y_wzsx_common_mean_de.parameters()) + list(self.y_wzsx_common_logv_de.parameters())+
        list(self.y_w0zsx_mean_de.parameters())+list(self.y_w0zsx_logv_de.parameters())
        +list(self.y_w1zsx_mean_de.parameters())+list(self.y_w1zsx_logv_de.parameters())+
        list(self.prior_mean.parameters())+list(self.logl.parameters())
        )

    ##########encoder
    def encoder_estALU_s_wx(self, w, x):
        common_mean_hidden=self.s_wx_common_mean_en(x)
        common_logv_hidden=self.s_wx_common_logv_en(x)
        s_w0x_mean = self.s_w0x_mean_en(common_mean_hidden)
        s_w0x_logv = self.s_w0x_logv_en(common_logv_hidden)
        s_w1x_mean = self.s_w1x_mean_en(common_mean_hidden)
        s_w1x_logv = self.s_w1x_logv_en(common_logv_hidden)
        s_wxz_mean, s_wxz_logv = torch.where(w.unsqueeze(dim=1)==1, s_w1x_mean, s_w0x_mean), torch.where(w.unsqueeze(dim=1)==1, s_w1x_logv, s_w0x_logv)
        # return s_wxz_mean.squeeze(), s_wxz_logv.squeeze().exp()
        return s_wxz_mean.squeeze(), torch.zeros_like(s_wxz_logv.squeeze()).exp()

        #修改恢复Z用到的参数
    def encoder_parma_z_xwsu(self,x,w,s,u):
        if s.dim() == 1:
            # 如果是一维，使用 unsqueeze 添加一个新维度
            xsu=torch.cat((x, s.unsqueeze(dim=1),u ),1)
        else:
            xsu = torch.cat((x, s, u), 1)
        #xsuw=torch.cat((x, s.unsqueeze(dim=1),u, w.unsqueeze(dim=1) ),1)

        common_mean_hidden = self.z_wxsu_common_mean_en(xsu)
        common_logv_hidden = self.z_wxsu_common_logv_en(xsu)
        z_w0xsu_mean=self.z_w0xsu_mean_en(common_mean_hidden)
        z_w0xsu_logv=self.z_w0xsu_logv_en(common_logv_hidden)
        z_w1xsu_mean=self.z_w1xsu_mean_en(common_mean_hidden)
        z_w1xsu_logv=self.z_w1xsu_logv_en(common_logv_hidden)

        z_mean, z_logv = torch.where(w.unsqueeze(dim=1) == 1, z_w1xsu_mean, z_w0xsu_mean), torch.where(w.unsqueeze(dim=1) == 1, z_w1xsu_logv, z_w0xsu_logv)
        # z_mean=self.z_wxsu_mean_en(xsuw)
        # z_logv=self.z_wxsu_logv_en(xsuw)

        return z_mean.squeeze(),z_logv.squeeze().exp()

    ###############decoder
    def decoder_parma_x_zu(self,z,u):
        if z.ndim == 1:
            # 变形为 (n, 1)
            z = z.unsqueeze(1)
        zu = torch.cat((z,u),1)
        x_mean=self.x_zu_mean_de(zu)
        x_logv=self.x_zu_logv_de(zu)
        return x_mean,x_logv.exp()

    def decoder_param_w_xz(self,x,z):
        if z.ndim == 1:
            # 变形为 (n, 1)
            z = z.unsqueeze(1)
        xz = torch.cat((x, z), 1)
        w_xz_p = torch.sigmoid(self.w_xz_p_de(xz))
        return w_xz_p

    def decoder_param_s_wxz(self, w, x, z):
        if z.ndim == 1:
            # 变形为 (n, 1)
            z = z.unsqueeze(1)
        xz = torch.cat((x, z), 1)

        common_mean_hidden = self.s_wxz_common_mean_de(xz)
        common_logv_hidden = self.s_wxz_common_logv_de(xz)
        s_w0xz_mean = self.s_w0xz_mean_de(common_mean_hidden)
        s_w0xz_logv = self.s_w0xz_logv_de(common_logv_hidden)
        s_w1xz_mean = self.s_w1xz_mean_de(common_mean_hidden)
        s_w1xz_logv = self.s_w1xz_logv_de(common_logv_hidden)
        s_wxz_mean, s_wxz_logv = torch.where(w.unsqueeze(dim=1)==1, s_w1xz_mean, s_w0xz_mean), torch.where(w.unsqueeze(dim=1)==1, s_w1xz_logv, s_w0xz_logv)
        return s_wxz_mean.squeeze(), s_wxz_logv.squeeze().exp()

    def decoder_param_y_wzsx(self, w, z, s, x):
        if z.ndim == 1:
            # 变形为 (n, 1)
            z = z.unsqueeze(1)
        if s.dim() == 1:
            # 如果是一维，使用 unsqueeze 添加一个新维度
            zsx = torch.cat((z, s.unsqueeze(dim=1), x), 1)
        else:
            zsx = torch.cat((z, s, x), 1)

        common_mean_hidden = self.y_wzsx_common_mean_de(zsx)
        common_logv_hidden = self.y_wzsx_common_logv_de(zsx)
        y_w0zsx_mean = self.y_w0zsx_mean_de(common_mean_hidden)
        y_w0zsx_logv = self.y_w0zsx_logv_de(common_logv_hidden)
        y_w1zsx_mean = self.y_w1zsx_mean_de(common_mean_hidden)
        y_w1zsx_logv = self.y_w1zsx_logv_de(common_logv_hidden)
        y_wzsx_mean, y_wzsx_logv = torch.where(w.unsqueeze(dim=1) == 1, y_w1zsx_mean, y_w0zsx_mean), torch.where(
            w.unsqueeze(dim=1) == 1, y_w1zsx_logv, y_w0zsx_logv)
        return y_wzsx_mean.squeeze(), y_wzsx_logv.squeeze().exp()


    ########先验p（z|U）的参数训练
    def prior_params(self, u):
        prior_mean = self.prior_mean(u)
        logl = self.logl(u)
        return prior_mean, logl.exp()


    def forward(self,x,u,s,w):

        prior_params = self.prior_params(u)

        #encoder
        s_params_en = self.encoder_estALU_s_wx(w, x)
        s_en = self.decoder_dist.sample(*s_params_en)

        z_params_en = self.encoder_parma_z_xwsu(x,w,s_en,u)
        z = self.encoder_dist.sample(*z_params_en)

        #decoder
        x_params_de=self.decoder_parma_x_zu(z,u)
        x_de=self.decoder_dist.sample(*x_params_de)

        w_param_de=self.decoder_param_w_xz(x,z)
        w_de=bernoulli.Bernoulli(w_param_de).sample()

        s_params_de=self.decoder_param_s_wxz(w,x,z)
        s_de=self.decoder_dist.sample(*s_params_de)

        y_params_de=self.decoder_param_y_wzsx(w,z,s_de,x)
        y_de=self.decoder_dist.sample(*y_params_de)
        if self.training:
            return  s_params_en,z_params_en, x_params_de, w_param_de, s_params_de, y_params_de, z, prior_params
        else:
            return s_params_en,z_params_en, x_params_de, w_param_de, s_params_de, y_params_de, z, prior_params


    def lossfun(self,x,u,w,s,y):

        s_params_en,z_params_en, x_params_de, w_param_de, s_params_de, y_params_de, z, prior_params=self.forward(x,u,s,w)

        log_pz_u=self.prior_dist.log_pdf(z,*prior_params)

        #decoder
        log_px_zu=self.decoder_dist.log_pdf(x,*x_params_de)

        neg_log_pw_xz=self.bce_loss(w_param_de,w.unsqueeze(dim=1)*1.0)

        log_ps_wzx = self.decoder_dist.log_pdf(s, *s_params_de)

        log_py_zsxw = self.decoder_dist.log_pdf(y, *y_params_de)

        # encoder
        log_qz_xu = self.encoder_dist.log_pdf(z, *z_params_en)#注意原式子中应该是z_xwsyu，这里是简化的encoder所以只有xu condition

        #log_qs_wx = self.encoder_dist.log_pdf(s, *s_params_en)#这个函数给exp训练了，obs数据集不具备对这个的可识别性，所以调整梯度时候不调整这个网络

        #ELBO = (log_pz_u + log_px_z - neg_log_pw_xz + log_ps_wzx + log_py_zsxw - log_qz_xu  +log_qs_wx).mean()  # 去掉log_qs_wx前
        ELBO = ( log_pz_u + log_px_zu - neg_log_pw_xz + log_ps_wzx + log_py_zsxw - log_qz_xu ).mean()  # 去掉log_qs_wx前
        #系数倍数全部去掉最后
        return ELBO



    #####根据输入估计潜在结果
    def infer_Potential(self,x,u,w):
        #z_params_en=self.encoder_parma_z_xu(x,u)


        s_params_en = self.encoder_estALU_s_wx(w, x)
        s_en, _ = s_params_en


        z_params_en=self.encoder_parma_z_xwsu(x,w,s_en,u)
        # z_params_en=self.encoder_parma_z_xwsu(x,w,s,u)
        z,_=z_params_en

        #decoder
        s_params_de = self.decoder_param_s_wxz(torch.ones_like(w), x, z)
        s_de1, _ = s_params_de

        s_params_de = self.decoder_param_s_wxz(torch.zeros_like(w), x, z)
        s_de0, _ = s_params_de

        y_params_de_1 = self.decoder_param_y_wzsx(torch.ones_like(w), z, s_de1, x)
        y_de_1, _ = y_params_de_1
        y_params_de_0 = self.decoder_param_y_wzsx(torch.zeros_like(w), z, s_de0, x)
        y_de_0, _ = y_params_de_0

        y_de_w = w * y_de_1 + (1 - w) * y_de_0

        return y_de_w


    def infer_Potential_Normalization_ver(self,x,u,ym,yv,w):#因为y是做过归一化的，所以需要recover
        s_params_en = self.encoder_estALU_s_wx(w, x)
        s_en, _ = s_params_en

        z_params_en = self.encoder_parma_z_xwsu(x, w, s_en, u)
        # z_params_en=self.encoder_parma_z_xwsu(x,w,s,u)
        z, _ = z_params_en

        # decoder
        s_params_de = self.decoder_param_s_wxz(torch.ones_like(w), x, z)
        s_de1, _ = s_params_de

        s_params_de = self.decoder_param_s_wxz(torch.zeros_like(w), x, z)
        s_de0, _ = s_params_de

        y_params_de_1 = self.decoder_param_y_wzsx(torch.ones_like(w), z, s_de1, x)
        y_de_1, _ = y_params_de_1
        y_params_de_0 = self.decoder_param_y_wzsx(torch.zeros_like(w), z, s_de0, x)
        y_de_0, _ = y_params_de_0

        y_de_w = w * y_de_1 + (1 - w) * y_de_0


        y_de_w_real = (y_de_w * yv.squeeze() + ym.squeeze())  # 反向normalization恢复

        return y_de_w_real


    def infer_Potential_Encoder_no_Estimate_S01(self, x, u, s, w):
        # z_params_en=self.encoder_parma_z_xu(x,u)

        z_params_en = self.encoder_parma_z_xwsu(x, w, s, u)
        # z_params_en=self.encoder_parma_z_xwsu(x,w,s,u)
        z, _ = z_params_en

        # decoder
        #W=1选这个
        s_params_de = self.decoder_param_s_wxz(torch.ones_like(w), x, z)
        s_de1, _ = s_params_de

        y_params_de_1 = self.decoder_param_y_wzsx(torch.ones_like(w), z, s_de1, x)
        y_de_1, _ = y_params_de_1

        #W=0选这个
        s_params_de = self.decoder_param_s_wxz(torch.zeros_like(w), x, z)
        s_de0, _ = s_params_de

        y_params_de_0 = self.decoder_param_y_wzsx(torch.zeros_like(w), z, s_de0, x)
        y_de_0, _ = y_params_de_0

        #根据W=1还是0选择
        y_de_w = w * y_de_1 + (1 - w) * y_de_0

        return y_de_w




    #实验组的forward，只是训练子网络中WX估计S0S1的encoder_estALU_s_wx
    def forward_exp(self,x,w):
        s_params_en = self.encoder_estALU_s_wx(w, x)

        return s_params_en

    def lossfun_exp(self,x,w,s):
        s_params_en=self.forward_exp(x,w)
        log_qs_wx = self.encoder_dist.log_pdf(s, *s_params_en)
        return log_qs_wx.mean()



    def culculateITE(self,x,u,s):
        return self.infer_Potential(x,u,w=torch.ones(x.shape[0]).to(self.device))-self.infer_Potential(x,u,w=torch.zeros(x.shape[0]).to(self.device))

    def culculateITE_AbsError(self,x,u,s,z):
        return torch.abs(self.infer_Potential(x,u,w=torch.ones(x.shape[0]).to(self.device))-self.infer_Potential(x,u,w=torch.zeros(x.shape[0]).to(self.device))-(7+2*torch.mean(x, axis=1)+3*torch.mean(z,axis=1)))

    def culculateITE_Error_2(self,x,u,s,z):
        return torch.square(self.infer_Potential(x,u,w=torch.ones(x.shape[0]).to(self.device))-self.infer_Potential(x,u,w=torch.zeros(x.shape[0]).to(self.device))-(7+2*torch.mean(x, axis=1)+3*torch.mean(z,axis=1)))


    def culculateITE_Nor(self, x, u,ym,yv):
        return self.infer_Potential_Normalization_ver(x, u, ym , yv , w=torch.ones(x.shape[0]).to(self.device)) - self.infer_Potential_Normalization_ver(x, u,ym,yv,w=torch.zeros(x.shape[0]).to(self.device))
#########################################################################################################5.17后写的文件不使用下面的方法来算评价指标，如果使用了这个计算的DGP是5.10之前的旧DGP的，所以新的dGP下的PEHE计算会出错
    def culculateITE_AbsError_Nor(self, x_nor, u_nor,z_nor,ym,yv,xm,xv,zm,zv):
        x=x_nor*xv.squeeze()+xm.squeeze()
        z=z_nor*zv.squeeze()+zm.squeeze()
        return torch.abs(self.infer_Potential_Normalization_ver(x_nor, u_nor, ym, yv, w=torch.ones(x_nor.shape[0]).to(self.device)) - self.infer_Potential_Normalization_ver(x_nor, u_nor, ym, yv,w=torch.zeros(x_nor.shape[0]).to(self.device)) - torch.from_numpy(DGP.Generate_Y(self,1,x.numpy(),1,z.numpy(),DGP.Generate_S(self,1,x.numpy(),1,z.numpy(),0),0)-DGP.Generate_Y(self,1,x.numpy(),0,z.numpy(),DGP.Generate_S(self,1,x.numpy(),0,z.numpy(),0),0)))
        #return torch.abs(self.infer_Potential_Normalization_ver(x_nor, u_nor,ym,yv, w=torch.ones(x_nor.shape[0])) - self.infer_Potential_Normalization_ver(x_nor, u_nor,ym,yv,w=torch.zeros(x_nor.shape[0])) - (7+0*torch.mean(x, axis=1)+7*torch.mean(z,axis=1)))

    def culculateITE_Error_2_Nor(self, x_nor, u_nor,z_nor,ym,yv,xm,xv,zm,zv):
        # x = x_nor * xv.squeeze() + xm.squeeze()
        # z = z_nor * zv.squeeze() + zm.squeeze()
        x = x_nor * xv + xm
        z = z_nor * zv + zm
        return torch.square(self.infer_Potential_Normalization_ver(x_nor, u_nor,ym,yv, w=torch.ones(x_nor.shape[0]).to(self.device)) - self.infer_Potential_Normalization_ver(x_nor, u_nor,ym,yv,w=torch.zeros(x_nor.shape[0]).to(self.device)) - torch.from_numpy(DGP.Generate_Y(self,1,x.numpy(),1,z.numpy(),DGP.Generate_S(self,1,x.numpy(),1,z.numpy(),0),0)-DGP.Generate_Y(self,1,x.numpy(),0,z.numpy(),DGP.Generate_S(self,1,x.numpy(),0,z.numpy(),0),0))  )

#############################################################################################################
    def culculateITE_Normalization_ver(self,x,u,ym,yv,realITEList):
        return self.infer_PO_simply(x, u, ym=ym, yv=yv, w=torch.ones(x.shape[0]).to(self.device)) - self.infer_PO_simply(x, u, ym=ym,yv=yv,w=torch.zeros(x.shape[0]).to(self.device))


#这个类放的是做数据处理的方法，主要是处理数据的归一化问题
class Data_Pretreatment():
    def normalization(x):#默认是troch变量
        X = x.clone()
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        X_standardized = (X - X_mean) / X_std
        return X_standardized, X_mean, X_std
    def normalization_m_s(x,X_mean,X_std):#指定均值方差后的归一化，用于统一
        X = x.clone()
        X_standardized = (X - X_mean) / X_std
        return X_standardized

    def recover_Normalization(self,X,X_mean,X_std):
        x = X.clone()
        X_recover=x*X_std+X_mean
        return x

    def  normalization_exp_and_obs(self,X_exp,X_obs):
        x_exp=X_exp.clone()
        x_obs=X_obs.clone()
        X_total=torch.cat((x_exp, x_obs), dim=0)
        #exp和obs的总体的mean和std
        total_mean=X_total.mean(dim=0)
        total_std=X_total.std(dim=0)
        #归一化两个变量
        x_exp_nor=(x_exp-total_mean)/total_std
        x_obs_nor=(x_obs-total_mean)/total_std
        return x_exp_nor,x_obs_nor,total_mean,total_std

class PreWhitener(nn.Module):
    """
    Data pre-whitener.
    """

    def __init__(self, data):
        super().__init__()
        with torch.no_grad():
            loc = data.mean(0)
            scale = data.std(0)
            scale[~(scale > 0)] = 1.0
            self.register_buffer("loc", loc)
            self.register_buffer("inv_scale", scale.reciprocal())

    def forward(self, data):
        return (data - self.loc) * self.inv_scale