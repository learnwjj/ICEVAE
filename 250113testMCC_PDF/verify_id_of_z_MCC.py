import argparse
import math
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns

from model.ICEVAE import ICEVAE,PreWhitener
from model.ICEVAE import Data_Pretreatment as DP

from evaluation_indicators.evaluate_fun import Evaluation_Instrument as evl_In
from data.Synthetic_data.Latent_Uncon_Linear_Version import dgp
import matplotlib.pyplot as plt
from test_MCC_dataset import dgp
from model.smalltool.MCC import mean_corr_coef



LOG_FOLDER = 'log/'
TENSORBOARD_RUN_FOLDER = 'runs/'
TORCH_CHECKPOINT_FOLDER = 'ckpt/'

def fixSeed(seednum):
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seednum)




if __name__ == '__main__':

    DP  = DP()
    evl_In=evl_In()
    num_experiments = 1

    PEHEList = []
    realATEList = []
    esti_ATEList = []


    for i in range(num_experiments):
        fixSeed(1)
        np.random.seed(1)
        print(f"Running experiment {i+1} with seed {i}")

        parser = argparse.ArgumentParser()

        parser.add_argument("--latent-dim", default=2, type=int)
        parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size (default 64)')
        parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs (default 20)')
        parser.add_argument('-g', '--hidden-dim', type=int, default=32, help='hidden dim of the networks (default 50)')
        parser.add_argument('--lr-exp', type=float, default=1e-3, help='learning rate (default 1e-5)')
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




        U_total, Z_total, X_total, W_total, S_total, Y_total, G_total, \
            U, Z, X, W, S, Y, \
            U_exp, Z_exp, X_exp, W_exp, S_exp, Y_exp, \
            Ite_List, Ite_List_exp=dgp()

        model=ICEVAE(U.shape[1], args.latent_dim, X.shape[1], 1, 1, 1,hidden_dim=args.hidden_dim ,device=device, layer_norm=False, activation_exit=False,layer_nums=1)
        optimizer_obs=torch.optim.Adam(model.train_parameters_obs(), lr=args.lr_obs)
        optimizer_exp=torch.optim.Adam(model.train_ens_parameters_exp(), lr=args.lr_exp)

        ste=time.time()

        print('setuptime:{}s'.format(ste-st))

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


        X_exp_Nor, X_Nor, X_mout, X_vout = DP.normalization_exp_and_obs(X_exp, X)
        U_exp_Nor, U_Nor, U_mout, U_vout = DP.normalization_exp_and_obs(U_exp, U)
        Y_exp_Nor, Y_Nor, Y_mout, Y_vout = DP.normalization_exp_and_obs(Y_exp,Y)
        Z_exp_Nor, Z_Nor, Z_mout, Z_vout = DP.normalization_exp_and_obs(Z_exp, Z)
        S_exp_Nor, S_Nor, S_mout, S_vout = DP.normalization_exp_and_obs(S_exp, S)

        test_size = 0.2
        val_size = 0.2
        X_Nor_train, X_Nor_test, U_Nor_train, U_Nor_test, Y_Nor_train, Y_Nor_test, W_train, W_test, S_Nor_train, S_Nor_test, Z_Nor_train, Z_Nor_test, Ite_List_train, Ite_List_test = \
            train_test_split(
                X_Nor, U_Nor, Y_Nor, W, S_Nor, Z_Nor, Ite_List, test_size=test_size, random_state=1
            )
        X_Nor_train, X_Nor_val, U_Nor_train, U_Nor_val, Y_Nor_train, Y_Nor_val, W_train, W_val, S_Nor_train, S_Nor_val, Z_Nor_train, Z_Nor_val, Ite_List_train, Ite_List_val = \
            train_test_split(
                X_Nor_train, U_Nor_train, Y_Nor_train, W_train, S_Nor_train, Z_Nor_train, Ite_List_train, test_size=val_size, random_state=1
            )



        #train dataset
        dset=TensorDataset(U_Nor_train,Z_Nor_train,X_Nor_train,W_train,S_Nor_train.squeeze(),Y_Nor_train.squeeze())
        train_loader=DataLoader(dset,shuffle=True,batch_size=args.batch_size)
        # test dataset
        dset_test=TensorDataset(U_Nor_test,Z_Nor_test,X_Nor_test,W_test,S_Nor_test.squeeze(),Y_Nor_test.squeeze())
        test_loader=DataLoader(dset_test,shuffle=True,batch_size=args.batch_size)
        # val dataset
        dset_val = TensorDataset(U_Nor_val,Z_Nor_val,X_Nor_val,W_val,S_Nor_val.squeeze(),Y_Nor_val.squeeze())
        val_loader = DataLoader(dset_val,shuffle=True,batch_size=args.batch_size)

        # exp dataset
        dset_exp = TensorDataset(U_exp_Nor, Z_exp_Nor, X_exp_Nor, W_exp, S_exp_Nor.squeeze(), Y_exp_Nor.squeeze())
        train_loader_exp = DataLoader(dset_exp, shuffle=True, batch_size=int(args.batch_size/4))

        realATE_test=np.mean(Ite_List_test)

        best_pehe = float('inf')
        best_model_params = None


        white=PreWhitener(X_Nor)

        it = 0

        ITE_PEHE = 0
        ATE_test = 0
        model.train()
        while it < args.epochs:

            ATE = 0
            ITEmeanloss = 0
            ITEloss2list = 0
            est = time.time()
            ATE_Ver2 = 0

            for iter, (u_exp, z_exp, x_exp, w_exp, s_exp, y_exp) in enumerate(train_loader_exp):
                x_exp=white(x_exp)

                optimizer_exp.zero_grad()

                loss_exp = - model.lossfun_exp(x_exp, w_exp, s_exp)

                loss_exp.backward()

                optimizer_exp.step()

            for iter, (u, z, x, w, s, y) in enumerate(train_loader):
                x = white(x)
                optimizer_obs.zero_grad()

                elbo = model.lossfun(x, u, w, s, y)

                (-1 * elbo).backward()

                optimizer_obs.step()

            it += 1
            eet = time.time()
            model.eval()
            model.train()


            ITE_esti_list_for_test=model.culculateITE_Nor(white(X_Nor_test),U_Nor_test,Y_mout,Y_vout)
            ATE_test=torch.mean(ITE_esti_list_for_test)
            ITE_PEHE=evl_In.pehe_normalization_ver(Ite_List_test,white(X_Nor_test),U_Nor_test,model.culculateITE_Nor,Y_mout,Y_vout)


            print(
                        'epoch {} done in: {}s;\texp_loss:{};\t-elbo_loss: {};\t'.
                        format(int(it), eet - est, loss_exp, -elbo))

        model.eval()
        et = time.time()
        print('training time: {}s'.format(et - ste))

        print('total time: {}s'.format(et - st))

        ITE_esti_list_for_test = model.culculateITE_Nor(X_Nor_test, U_Nor_test, Y_mout, Y_vout)
        ATE_test = torch.mean(ITE_esti_list_for_test)
        ITE_PEHE = evl_In.pehe_normalization_ver(Ite_List_test, X_Nor_test, U_Nor_test, model.culculateITE_Nor, Y_mout,
                                                 Y_vout)
        PEHEList.append(ITE_PEHE.detach().numpy())
        realATEList.append(realATE_test)
        esti_ATEList.append(ATE_test.detach().numpy())


        _, _, _, _, _, _, reover_Z_Nor_train, _ = model.forward(X_Nor_train, U_Nor_train, S_Nor_train, W_train)


        true_Z=Z_Nor_test

        model.eval()
        with torch.no_grad():
            _,_, _, _, _, _, learn_Z, _=model.forward(X_Nor_test, U_Nor_test,S_Nor_test,W_test)

            learn_Z=(learn_Z- torch.mean(learn_Z,axis=0))/ torch.std(learn_Z, axis=0)
            true_Z=(true_Z- torch.mean(true_Z,axis=0))/ torch.std(true_Z, axis=0)


            mcc_values = []
            combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for i, j in combinations:
                mcc_value = mean_corr_coef(true_Z[:, i].unsqueeze(1).numpy(), learn_Z[:, j].unsqueeze(1).numpy(), method='rdc')
                mcc_values.append(mcc_value)
                print(mcc_value)

            # plt.figure(figsize=(12, 12))
            # fontdict = {'fontsize': 14, 'fontweight': 'bold'}
            plt.figure(figsize=(12, 12))
            fontdict = {'fontsize': 30, 'fontweight': 'bold'}
            fontdict_axis = {'fontsize': 35, 'fontweight': 'bold'}
            font_size = 25
            for idx, (i, j) in enumerate(combinations):
                plt.subplot(2, 2, idx + 1)
                plt.scatter(true_Z[:, i].numpy(), learn_Z[:, j].numpy(), alpha=0.6, color='blue')
                #plt.title(f"True Z dim {i + 1} vs Learned Z dim {j + 1} mcc value{mcc_values[i+j*2]}\n")
                plt.xlabel(f"True Z dim {i + 1}",fontdict = fontdict)
                plt.ylabel(f"Learned Z dim {j + 1}",fontdict = fontdict)
                plt.text(0.70, 0.90, f'MCC: {mcc_values[i+j*2]:.4f}', transform=plt.gca().transAxes,fontdict = fontdict)
                plt.grid(False)

            plt.tight_layout()
            plt.show()



            i = 0
            j = 0
            plt.figure(figsize=(6, 6))
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.scatter(true_Z[:, i].numpy(), learn_Z[:, j].numpy(), alpha=0.6, color='blue')
            # plt.title(f"True Z dim {i + 1} vs Learned Z dim {j + 1} mcc value{mcc_values[i+j*2]}\n")
            plt.xlabel(f"True Z dim {i + 1}", fontdict=fontdict_axis)
            plt.ylabel(f"Estimated Z dim {j + 1}", fontdict=fontdict_axis)
            plt.text(0.35, 0.90, f'MCC: {mcc_values[i + j * 2]:.4f}', transform=plt.gca().transAxes, fontdict=fontdict)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig('MCC_1.pdf')
            plt.show()

            i = 0
            j = 1
            plt.figure(figsize=(6, 6))
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.scatter(true_Z[:, i].numpy(), learn_Z[:, j].numpy(), alpha=0.6, color='blue')
            # plt.title(f"True Z dim {i + 1} vs Learned Z dim {j + 1} mcc value{mcc_values[i+j*2]}\n")
            plt.xlabel(f"True Z dim {i + 1}", fontdict=fontdict_axis)
            plt.ylabel(f"Estimated Z dim {j + 1}", fontdict=fontdict_axis)
            plt.text(0.35, 0.90, f'MCC: {mcc_values[i + j * 2]:.4f}', transform=plt.gca().transAxes, fontdict=fontdict)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig('MCC_2.pdf')
            plt.show()

            i = 1
            j = 0
            plt.figure(figsize=(6, 6))
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.scatter(true_Z[:, i].numpy(), learn_Z[:, j].numpy(), alpha=0.6, color='blue')
            # plt.title(f"True Z dim {i + 1} vs Learned Z dim {j + 1} mcc value{mcc_values[i+j*2]}\n")
            plt.xlabel(f"True Z dim {i + 1}", fontdict=fontdict_axis)
            plt.ylabel(f"Estimated Z dim {j + 1}", fontdict=fontdict_axis)
            plt.text(0.35, 0.90, f'MCC: {mcc_values[i + j * 2]:.4f}', transform=plt.gca().transAxes, fontdict=fontdict)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig('MCC_3.pdf')
            plt.show()

            i = 1
            j = 1
            plt.figure(figsize=(6, 6))
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.scatter(true_Z[:, i].numpy(), learn_Z[:, j].numpy(), alpha=0.6, color='blue')
            # plt.title(f"True Z dim {i + 1} vs Learned Z dim {j + 1} mcc value{mcc_values[i+j*2]}\n")
            plt.xlabel(f"True Z dim {i + 1}", fontdict=fontdict_axis)
            plt.ylabel(f"Estimated Z dim {j + 1}", fontdict=fontdict_axis)
            plt.text(0.35, 0.90, f'MCC: {mcc_values[i + j * 2]:.4f}', transform=plt.gca().transAxes, fontdict=fontdict)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig('MCC_4.pdf')
            plt.show()







