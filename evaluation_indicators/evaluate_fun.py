import torch
class Evaluation_Instrument():
    def __int__(self):
        return

    def pehe_normalization_ver(self,ite_groundtruth_list,X,U,ite_estimate_model,Y_mean,Y_std,device='cpu'):
        ite_estimate_list=ite_estimate_model(X,U,Y_mean,Y_std)
        pehe_test = torch.mean(torch.square(torch.from_numpy(ite_groundtruth_list).to(device) - ite_estimate_list))
        return pehe_test

    def pehe(self,ite_groundtruth_list,X,U,ite_estimate_model,device='cpu'):
        ite_estimate_list=ite_estimate_model(X,U)
        pehe_test = torch.mean(torch.var(torch.from_numpy(ite_groundtruth_list).to(device) - ite_estimate_list))
        return pehe_test
