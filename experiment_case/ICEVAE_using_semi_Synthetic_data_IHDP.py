"""
our Synthetic dataset + ICEVAE model write by ourself not based on pyro 5.15
"""
import argparse
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model.ICEVAE import ICEVAE
from model.ICEVAE import Data_Pretreatment as DP
from data.semi_Synthetic_data_IHDP import newDGP
from evaluation_indicators.evaluate_fun import Evaluation_Instrument as evl_In



LOG_FOLDER = 'log/'
TENSORBOARD_RUN_FOLDER = 'runs/'
TORCH_CHECKPOINT_FOLDER = 'ckpt/'

def fixSeed(seednum):
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seednum)

if __name__ == '__main__':
    DGP = newDGP()
    DP  = DP()
    evl_In=evl_In()
    num_experiments = 5

    PEHEList = []
    realATEList = []
    esti_ATEList = []

    for i in range(num_experiments):
        fixSeed(i)
        np.random.seed(i)
        print(f"Running experiment {i+1} with seed {i}")
        parser = argparse.ArgumentParser()
        parser.add_argument("--latent-dim", default=10, type=int)
        parser.add_argument('-b', '--batch-size', type=int, default=50, help='batch size (default 64)')
        parser.add_argument('-e', '--epochs', type=int, default=200, help='number of epochs (default 20)')
        parser.add_argument('-g', '--hidden-dim', type=int, default=64, help='hidden dim of the networks (default 50)')
        parser.add_argument('--lr-exp', type=float, default=1e-4, help='learning rate (default 1e-5)')
        parser.add_argument('--lr-obs', type=float, default=1e-4, help='learning rate (default 1e-5)')
        parser.add_argument('-c', '--cuda', action='store_true', default=False, help='train on gpu')
        parser.add_argument('-p', '--preload-gpu', action='store_true', default=False, dest='preload',
                            help='preload data on gpu for faster training.')
        args = parser.parse_args()

        print(args)

        torch.set_default_tensor_type(torch.DoubleTensor)

        st = time.time()

        device = torch.device('cuda' if args.cuda else 'cpu')

        print('training on {}'.format(torch.cuda.get_device_name(device) if args.cuda else 'cpu'))




        U, Z, X, W, S, Y, U_exp, Z_exp, X_exp, W_exp, S_exp, Y_exp,Ite_List, real_ATE_DGPCUL,Ite_List_exp, real_ATE_DGPCUL_exp = DGP.GenerateDataset_STDmethod(seed=i)

        model=ICEVAE(1, args.latent_dim, X.shape[1], 1, 7, 1,hidden_dim=args.hidden_dim, device=device,layer_norm=False,activation_exit=True,layer_nums=2,activation='leaky_relu',bin_using_sigmoid=True)
        optimizer_obs=torch.optim.Adam(model.train_parameters_obs(), lr=args.lr_obs)
        optimizer_exp=torch.optim.Adam(model.train_ens_parameters_exp(), lr=args.lr_exp)


        ste=time.time()

        print('setuptime:{}s'.format(ste-st))

        model.train()

        X = X.to(device=device, non_blocking=True)
        U = U.to(device=device, non_blocking=True)
        W = W.to(device=device, non_blocking=True)
        Y = Y.to(device=device, non_blocking=True)
        Z = Z.to(device=device, non_blocking=True)
        S = S.to(device=device, non_blocking=True)

        X_exp = X_exp.to(device=device, non_blocking=True)
        U_exp = U_exp.to(device=device, non_blocking=True)
        W_exp = W_exp.to(device=device, non_blocking=True)
        Y_exp = Y_exp.to(device=device, non_blocking=True)
        Z_exp = Z_exp.to(device=device, non_blocking=True)
        S_exp = S_exp.to(device=device, non_blocking=True)



        X_exp_Nor,X_Nor,X_mout,X_vout=DP.normalization_exp_and_obs(X_exp, X)
        U_exp_Nor, U_Nor, U_mout, U_vout = DP.normalization_exp_and_obs(U_exp, U)
        Y_exp_Nor, Y_Nor, Y_mout, Y_vout = DP.normalization_exp_and_obs(Y_exp,Y)
        Z_exp_Nor, Z_Nor, Z_mout, Z_vout = DP.normalization_exp_and_obs(Z_exp, Z)
        S_exp_Nor, S_Nor, S_mout, S_vout = DP.normalization_exp_and_obs(S_exp, S)


        test_size = 0.2

        X_Nor_train, X_Nor_test, U_Nor_train, U_Nor_test, Y_Nor_train, Y_Nor_test, W_train, W_test, S_Nor_train, S_Nor_test, Z_Nor_train, Z_Nor_test, Ite_List_train, Ite_List_test = \
            train_test_split(
                X_Nor, U_Nor, Y_Nor, W, S_Nor, Z_Nor, Ite_List, test_size=test_size, random_state=i
            )

        #train dataset
        dset=TensorDataset(U_Nor_train,Z_Nor_train,X_Nor_train,W_train,S_Nor_train.squeeze(),Y_Nor_train.squeeze())#前面nor时候为了能够norma加了unsqueeze，所以这里要把S，Y压缩回去
        train_loader=DataLoader(dset,shuffle=True,batch_size=args.batch_size)
        # test dataset
        dset_test=TensorDataset(U_Nor_test,Z_Nor_test,X_Nor_test,W_test,S_Nor_test.squeeze(),Y_Nor_test.squeeze())#前面nor时候为了能够norma加了unsqueeze，所以这里要把S，Y压缩回去
        test_loader=DataLoader(dset_test,shuffle=True,batch_size=args.batch_size)
        # exp dataset
        dset_exp = TensorDataset(U_exp_Nor, Z_exp_Nor, X_exp_Nor, W_exp, S_exp_Nor.squeeze(), Y_exp_Nor.squeeze())  #实验组数据集
        train_loader_exp = DataLoader(dset_exp, shuffle=True, batch_size=int(args.batch_size/4))


        realATE_test=np.mean(Ite_List_test)



        it = 0
        ITE_PEHE=0
        ATE_test=0

        while it < args.epochs:

            ATE = 0
            ITEmeanloss = 0
            ITEloss2list = 0
            est = time.time()
            ATE_Ver2 = 0

            for iter, (u_exp, z_exp, x_exp, w_exp, s_exp, y_exp) in enumerate(train_loader_exp):
                optimizer_exp.zero_grad()

                loss_exp = - model.lossfun_exp(x_exp, w_exp, s_exp)

                loss_exp.backward()

                optimizer_exp.step()

            for iter, (u, z, x, w, s, y) in enumerate(train_loader):

                optimizer_obs.zero_grad()

                elbo = model.lossfun(x, u, w, s, y)

                (-1 * elbo).backward()

                optimizer_obs.step()

            it += 1
            eet = time.time()


            ITE_esti_list_for_test=model.culculateITE_Nor(X_Nor_test,U_Nor_test,Y_mout,Y_vout)
            ATE_test=torch.mean(ITE_esti_list_for_test)
            ITE_PEHE=evl_In.pehe_normalization_ver(Ite_List_test,X_Nor_test,U_Nor_test,model.culculateITE_Nor,Y_mout,Y_vout)


            print(
                        'epoch {} done in: {}s;\texp_loss:{};\t-elbo_loss: {};\trealATE_test:{};\tAstimateATE:{};\tITE_PEHE:{}'.
                        format(int(it), eet - est, loss_exp, -elbo,  realATE_test, ATE_test,ITE_PEHE))

        et = time.time()
        model.eval()
        print('training time: {}s'.format(et - ste))

        print('total time: {}s'.format(et - st))

        PEHEList.append(ITE_PEHE.detach().numpy())
        realATEList.append(realATE_test)
        esti_ATEList.append(ATE_test.detach().numpy())



    def calculate_mse(ground_truth, prediction):
        mse = (ground_truth - prediction) ** 2
        return mse


    PEHE_mean = np.mean(PEHEList)

    PEHE_std = np.std(PEHEList)

    print("total PEHE:", PEHE_mean, "+-", PEHE_std)

    mselist = calculate_mse(np.array(realATEList), np.array(esti_ATEList))
    MSEmean = np.mean(mselist)
    MSEstd = np.std(mselist)

    print("total MSE:", MSEmean, "+-", MSEstd)






