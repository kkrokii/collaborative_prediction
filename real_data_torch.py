import numpy as np
import pandas as pd
import time
import argparse
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import os
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_X_y_and_tensorize(df, y_column):
    return torch.Tensor( df.drop(columns=[y_column]).values ).unsqueeze(0).to(device), torch.Tensor( df[[y_column]].values ).reshape(1,-1,1).to(device)     # batch_size(=1) x n x p, batch_size(=1) x n x 1

class DataSplitter:
    def split_train_estimation_test(self, df, train=0.4, estimation=0.4, test=0.2):
        total_samples=df.shape[0]                                                       # df : n x p
        train = int(total_samples * train)
        estimation = int(total_samples * estimation)
        test = int(total_samples * test)
        df = df.sample(frac=1)
        training_data = df[:train]
        estimation_data = df[train : train+estimation]
        test_data = df[train+estimation : train+estimation+test]
        return training_data, estimation_data, test_data                                # df, df, df



class Model:
    def __init__(self, alpha_list, bound_type=2):
        self.alpha_list = torch.Tensor(alpha_list).to(device)
        self.bound_type = bound_type
        self.beta_1_hat = None
        self.beta_2_hat = None
        self.beta_c_hat = None
    def inverse_of_covariance(self, X):
        return torch.linalg.inv(X.transpose(-1,-2) @ X)
    def coefficient(self, cov_inv, X, y):
        return cov_inv @ X.transpose(-1,-2) @ y
    def trace(self, M):
        tr = torch.zeros(M.shape[0])
        for i in range(len(tr)):
            tr[i] = torch.trace(M[i])
        return tr.to(device)
    def bound_constants(self, n1, n2, p, X1, y1, X2, y2, W, BTB, lhs_const, gamma = 0.9):
        """
        Xi : batch_size x ni x p,   yi : batch_size x ni x 1
        """
        Xc = torch.cat((X1, X2), dim=-2)                                                        # batch_size x (n1+n2) x p
        yc = torch.cat((y1, y2), dim=-2)                                                        # batch_size x (n1+n2) x 1
        X0 = torch.cat((torch.cat((X1, torch.zeros_like(X1)), dim=-1), torch.cat((torch.zeros_like(X2), X2), dim=-1)), dim=-2)   # batch_size x (n1+n2) x 2p
        y0 = yc
        
        cov_1_inv = self.inverse_of_covariance(X1)                                              # batch_size x p x p
        cov_2_inv = self.inverse_of_covariance(X2)
        cov_c_inv = self.inverse_of_covariance(Xc)  
        cov_0_inv = torch.cat((torch.cat((cov_1_inv, torch.zeros_like(cov_1_inv)), dim=-1), torch.cat((torch.zeros_like(cov_2_inv), cov_2_inv), dim=-1)), dim=-2)   # batch_size x 2p x 2p
        beta_1_hat = self.coefficient(cov_1_inv, X1, y1)                                        # batch_size x p x 1
        beta_2_hat = self.coefficient(cov_2_inv, X2, y2)
        beta_c_hat = self.coefficient(cov_c_inv, Xc, yc)
        self.beta_1_hat = beta_1_hat
        self.beta_2_hat = beta_2_hat
        self.beta_c_hat = beta_c_hat
        
        DTD = torch.linalg.inv( cov_1_inv + cov_2_inv )
        
        sigma_c_2_hat = ( torch.sum((y1 - X1 @ beta_c_hat)**2, dim=-2) + torch.sum((y2 - X2 @ beta_c_hat)**2, dim=-2) ) / (n1+n2-p)
        sigma_c_2_hat = sigma_c_2_hat.squeeze()                                                 # (batch_size, )
        
        C = torch.cat((torch.eye(p), -torch.eye(p)), dim=1).to(device)
        Sigma = X0 @ cov_0_inv @ C.transpose(0,1) @ BTB @ C @ cov_0_inv @ X0.transpose(-1,-2)   # batch_size x (n1+n2) x (n1+n2)
        norm_Sigma = torch.linalg.matrix_norm(Sigma, ord=2).to(device)                          # (batch_size, )

        A = X0 @ cov_0_inv @ C.transpose(0,1)                                                   # batch_size x (n1+n2) x p
        M = A @ self.inverse_of_covariance(A) @ A.transpose(-1,-2)                              # batch_size x (n1+n2) x (n1+n2)

        e = torch.exp(torch.Tensor([1])).to(device)
        if self.bound_type==1:
            const1 = (n1+n2-p)*sigma_c_2_hat + 2 * (y0.transpose(-1,-2) @ M @ y0).squeeze()
            const2 = self.trace(Sigma)
            const3 = 2*torch.sqrt(self.trace(Sigma @ Sigma))
            const4 = 2*norm_Sigma
            const5 = (1-gamma)**(2/(n1+n2-p)) * (n1+n2-p)/2/e -2*(n1+n2-p)
            const6 = ( -4*torch.sqrt(torch.Tensor([n1+n2-p])) ).to(device)
            const7 = 2*(beta_1_hat - beta_2_hat).transpose(-1,-2) @ BTB @ (beta_1_hat - beta_2_hat)
            rhs_bound_ft_const_list = [const1, const2, const3, const4, const5, const6, const7]
        else:
            const21 = (beta_1_hat-beta_2_hat).transpose(-1,-2) @ BTB @ (beta_1_hat-beta_2_hat)
            const22 = self.trace(Sigma)
            const23 = 2*torch.sqrt(self.trace(Sigma@Sigma))
            const24 = 2*norm_Sigma
            
            const25 = ( torch.log(torch.Tensor([4/(1+1-gamma)])) ).to(device)
            const26 = torch.sqrt(const25) + 1/2/torch.sqrt(const25)
            const27 = torch.sqrt((n1+n2-p)*sigma_c_2_hat)
            
            const28 = const27 * ( torch.sqrt((beta_1_hat-beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat-beta_2_hat)) ).squeeze()
            const29 = -(n1+n2-p)/8*(2-gamma)/torch.sqrt(e*const25)
            const32 = (beta_1_hat-beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat-beta_2_hat)
            const31 = torch.sqrt(torch.Tensor([2*(n1+n2-p)])).to(device) * torch.sqrt(const32) * const26
            rhs_bound_ft_const_list = [const21, const22, const23, const24, const26, const27, const28, const29, const31, const32]
            
        const11 = torch.trace(W @ lhs_const)
        const12 = (n1+n2-p) * sigma_c_2_hat - ( 2*(beta_1_hat - beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat - beta_2_hat) ).squeeze()
        const14 = torch.Tensor([n1+n2-p]).to(device)
        const13 = -4/torch.sqrt(const14) * ( (beta_1_hat - beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat - beta_2_hat) ).squeeze()
        const15 = 2*torch.sqrt(const14)
        const16 = 4/torch.sqrt(const14)
        lhs_bound_ft_const_list = [const11, const12, const13, const14, const15, const16]

        for i in range(len(rhs_bound_ft_const_list)):
            rhs_bound_ft_const_list[i] = rhs_bound_ft_const_list[i].reshape(-1,1).to(device)
        for i in range(len(lhs_bound_ft_const_list)):
            lhs_bound_ft_const_list[i] = lhs_bound_ft_const_list[i].reshape(-1,1).to(device)
        return rhs_bound_ft_const_list, lhs_bound_ft_const_list                     # constants are either batch_size x 1 or 1 x 1
    
    def rhs_bound_ft1(self, const_list):
        return 2*const_list[0]*(const_list[1] + const_list[2]*torch.sqrt(self.alpha_list) + const_list[3]*self.alpha_list) / (const_list[4] + const_list[5]*torch.sqrt(self.alpha_list) -4*self.alpha_list) + const_list[6]         # batch_size x len(alpha_list)
    def lhs_bound_ft(self, const_list):
        return const_list[0] * (const_list[1] + const_list[2]*torch.sqrt(self.alpha_list)) / (const_list[3] + const_list[4]*torch.sqrt(self.alpha_list) + 2*self.alpha_list) / (3 + const_list[5]*torch.sqrt(self.alpha_list))
    def rhs_bound_ft2(self, const_list):
        const = ( n1+n2-p + 2*torch.sqrt((n1+n2-p)*self.alpha_list) + 2*self.alpha_list ).reshape(1,-1)
        const_rhs1 = const_list[5]*torch.sqrt(const)
        const_rhs2 = const_list[6]
        const_rhs3 = const_list[7] - torch.sqrt(torch.Tensor([2*(n1+n2-p)]).to(device)) * const_list[4] * torch.sqrt(const) + const
        const_rhs4 = const_list[8] - 2*torch.sqrt(const_list[9])*torch.sqrt(const)
        const_rhs5 = const_list[9]
        sigma_bound = (const_rhs1-const_rhs4 + torch.sqrt( (const_rhs1-const_rhs4)**2 -4*const_rhs3*(const_rhs5-const_rhs2) ) ) / 2 / const_rhs3
        return (torch.sqrt(const_list[0]) + torch.sqrt(const_list[1] + const_list[2]*torch.sqrt(self.alpha_list) + const_list[3]*self.alpha_list) * sigma_bound )**2
    
    def get_bound_list(self, n1, n2, p, X1, y1, X2, y2, W, BTB, lhs_const, gamma = 0.9):
        rhs_bound_const_list, lhs_bound_const_list = self.bound_constants(n1, n2, p, X1, y1, X2, y2, W, BTB, lhs_const, gamma)
        rhs_bound_list = self.rhs_bound_ft1(rhs_bound_const_list) if self.bound_type==1 else self.rhs_bound_ft2(rhs_bound_const_list)
        lhs_bound_list = self.lhs_bound_ft(lhs_bound_const_list)
        return rhs_bound_list, lhs_bound_list                   # batch_size x len(alpha_list), batch_size x len(alpha_list)


class DataGenerator:
    def __init__(self, y_column):
        self.y_column = y_column
        
    def generate_bootstraps(self, df, n, samples = 1000):
        p = df.shape[-1]-1                                                           # exclude y_column
        X_bootstrap = torch.zeros(size=(samples, n, p)).to(device)
        y_bootstrap = torch.zeros(size=(samples, n, 1)).to(device)
        for i in range(samples):
            bootstrap = df.sample(n, replace=True)                                   # df with y_column included
            X, y = split_X_y_and_tensorize(df=bootstrap, y_column=self.y_column)            # 1 x n x p, 1 x n x 1 
            X_bootstrap[i] = X
            y_bootstrap[i] = y
        return X_bootstrap, y_bootstrap                                                     # samples x n x p


class Estimator(DataGenerator):            # bootstrap for samples?
    def __init__(self, estimation_data, y_column):
        self.data = estimation_data                                                         # estimation_data : df;     n x p
        self.y_column = y_column
        
    def mean_Z1(self, n1, n2, samples = 1000):
        X1_list, _ = self.generate_bootstraps(self.data, n1, samples)
        X2_list, _ = self.generate_bootstraps(self.data, n2, samples)
        cov_X1_list = X1_list.transpose(1,2) @ X1_list                                                     # samples x p x p
        cov_X2_list = X2_list.transpose(1,2) @ X2_list                                                     # samples x p x p
        Z1 = torch.sum( torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X1_list , dim=0 )
        return Z1 / samples

    def mean_Z1TWZ1(self, n1, n2, W, samples = 1000):
        X1_list, _ = self.generate_bootstraps(self.data, n1, samples)
        X2_list, _ = self.generate_bootstraps(self.data, n2, samples)
        cov_X1_list = X1_list.transpose(1,2) @ X1_list
        cov_X2_list = X2_list.transpose(1,2) @ X2_list
        Z1_list = torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X1_list
        Z1TWZ1 = torch.sum( Z1_list.transpose(1,2) @ W @ Z1_list, dim=0 )
        return Z1TWZ1 / samples
        
    def BTB(self, n1, n2, pi, W, samples = 1000):
        mean_Z1 = self.mean_Z1(n1, n2, samples = 1000)
        return self.mean_Z1TWZ1(n1, n2, W, samples) + pi * (W - mean_Z1.transpose(0,1) @ W - W @ mean_Z1)

    def mean_cov_inv(self, n, samples = 1000):
        X_list, _ = self.generate_bootstraps(self.data, n, samples)
        XTX_list = X_list.transpose(1,2) @ X_list
        XTX_inv = torch.sum( torch.linalg.inv(XTX_list), dim=0 )
        return XTX_inv / samples    
    
    def mean_cov(self, n, samples = 1000):
        X_list, _ = self.generate_bootstraps(self.data, n, samples)
        XTX_list = X_list.transpose(1,2) @ X_list
        XTX = torch.sum( XTX_list, dim=0 )
        return XTX / samples    
    
    def lhs_const(self, n1, n2, samples = 1000):
        return self.mean_cov_inv(n1, samples) + self.mean_cov_inv(n2, samples) - self.mean_cov_inv(n1+n2, samples)



class GTGenerator(DataGenerator):
    # does not account for noise_level
    def __init__(self, estimation_data1, estimation_data2, y_column):
        self.data1 = estimation_data1
        self.data2 = estimation_data2
        self.y_column = y_column
        
    def out_of_sample_error(self, beta_hat, batch_size, distribution=1, samples=1000):
        """
        beta_hat: batch_size x p x 1
        """
        df = self.data1 if distribution==1 else self.data2
        X0, y0 = self.generate_bootstraps(df = df, samples=batch_size, n=samples)                                    # batch_size x samples x p
        return (torch.sum((y0 - X0 @ beta_hat)**2, dim=-2)/samples).squeeze()                               # (batch_size, )

    def out_of_sample_error_combined(self, beta_c_hat, pi, batch_size, samples=2000):
        from_distr1 = int(round(pi*samples))
        X0_1, y0_1 = self.generate_bootstraps(df=self.data1, samples = batch_size, n=from_distr1,)                         # batch_size x from_distr1 x p
        X0_2, y0_2 = self.generate_bootstraps(df=self.data2, samples = batch_size, n=samples-from_distr1)                  # batch_size x (samples-from_distr1) x p
        return ( torch.sum((y0_1 - X0_1 @ beta_c_hat)**2, dim=-2) + torch.sum((y0_2 - X0_2 @ beta_c_hat)**2, dim=-2) ).squeeze()/samples      # (batch_size, )

    def gt_combine(self, beta1_hat, beta2_hat, beta_c_hat, pi, batch_size, samples=1000):
        error1 = self.out_of_sample_error(beta1_hat, batch_size, distribution=1, samples=samples)
        error2 = self.out_of_sample_error(beta2_hat, batch_size, distribution=2, samples=samples)
        error_combined = self.out_of_sample_error_combined(beta_c_hat, pi, batch_size, samples=samples*2)
        gt = error1+error2 > error_combined   # True: should combine;     False: should not combine       # (batch_size, )
        return gt.to(device)



def pairwise_comparison(n1, n2, p, df1, df2, y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, bound_type=2):
    tick = time.time()
    
    data_splitter = DataSplitter()
    train_data1, estimation_data1, test_data1 = data_splitter.split_train_estimation_test(df1, train = train_ratio, estimation = estimation_ratio, test = test_ratio)
    train_data2, estimation_data2, test_data2 = data_splitter.split_train_estimation_test(df2, train = train_ratio, estimation = estimation_ratio, test = test_ratio)
    # need to consider these as distr;      further bootstrap from each distr
    
    test_data = pd.concat([test_data1, test_data2])
    W_estimator = Estimator(test_data, y_column)
    W = W_estimator.mean_cov(n=1, samples = 1000)
    pi = df1.shape[-2] / ( df1.shape[-2] + df2.shape[-2] )
    
    estimation_data = pd.concat([estimation_data1, estimation_data2])
    parameter_estimator = Estimator(estimation_data, y_column)
    BTB = parameter_estimator.BTB(n1, n2, pi, W, samples = 1000)
    lhs_const = parameter_estimator.lhs_const(n1, n2, samples = 1000)
    print(f"\t constants estimated, time: {time.time()-tick}")
    
    data_generator = DataGenerator(y_column)
    batch_size=1000
    
    min_alpha, max_alpha = (2,10)
    alpha_list = torch.arange(min_alpha, max_alpha, 0.01)
    model = Model(alpha_list, bound_type)
    
    X1, y1 = data_generator.generate_bootstraps(df=train_data1, samples=batch_size, n=n1)     # batch_size x n1 x p, batch_size x n1 x 1
    X2, y2 = data_generator.generate_bootstraps(df=train_data2, samples=batch_size, n=n2)     # batch_size x n1 x p, batch_size x n1 x 1
    
    gt_generator = GTGenerator(train_data1, train_data2, y_column)
    rhs_bound_list, lhs_bound_list = model.get_bound_list(n1, n2, p, X1, y1, X2, y2, W, BTB, lhs_const, gamma=0.9)      # batch_size x len(alpha_list), batch_size x len(alpha_list)
    gt_list = gt_generator.gt_combine(model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, pi, batch_size=1, samples=1000)     # (batch_size, )
    gt_list = gt_list.reshape(-1,1)                      # batch_size x 1 
    
    combine = lhs_bound_list > rhs_bound_list
    correct_list = torch.sum( combine == gt_list, dim=0 )
    
    optimal_alpha = alpha_list[torch.argmax(correct_list)]
    alpha_list = torch.Tensor([optimal_alpha]).to(device)
    model = Model(alpha_list, bound_type)
    
    print(f"\t hyperparameter tuned, time: {time.time()-tick}, alpha: {optimal_alpha}")
    
    X1, y1 = data_generator.generate_bootstraps(df=test_data1, samples=batch_size, n=n1)     # batch_size x n1 x p, batch_size x n1 x 1
    X2, y2 = data_generator.generate_bootstraps(df=test_data2, samples=batch_size, n=n2)     # batch_size x n1 x p, batch_size x n1 x 1
    
    gt_generator = GTGenerator(test_data1, test_data2, y_column)
    rhs_bound_list, lhs_bound_list = model.get_bound_list(n1, n2, p, X1, y1, X2, y2, W, BTB, lhs_const, gamma=0.9)      # batch_size x len(alpha_list), batch_size x len(alpha_list)
    gt_list = gt_generator.gt_combine(model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, pi, batch_size=1, samples=1000)     # (batch_size, )
    gt_list = gt_list.reshape(-1,1)                      # batch_size x 1 
    
    combine = lhs_bound_list > rhs_bound_list
    correct = torch.sum( combine == gt_list )
    
    print(f"\t main ft finished, time: {time.time()-tick}")
    
    return correct, optimal_alpha, (rhs_bound_list, lhs_bound_list, gt_list)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='output.pd'
    )
    parser.add_argument(
        '--bound_type',
        type=int,
        default=2
    )
    parser.add_argument(
        '--y_column',
        type=str,
        default=None
    )
    parser.add_argument(
        '--input_file1',
        type=str,
        default=None
    )
    parser.add_argument(
        '--input_file2',
        type=str,
        default=None
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Creating {args.output_dir}")
    
    tick = time.time()
    
    n1=50
    n2=50
    df1 = pd.read_csv(args.input_file1)
    df2 = pd.read_csv(args.input_file2)
    p=df1.shape[1]-1
    
    print(f"main ft begins")
    result = pairwise_comparison(n1, n2, p, df1, df2, args.y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, bound_type=args.bound_type)
    
    output = pd.DataFrame.from_dict({
        'n1': [n1], 
        'n2': [n2], 
        'p': [p], 
        'correct': [result[0].cpu()],
        'alpha': [result[1].cpu()],
        'rhs_bound_list': [result[2][0].cpu()],
        'lhs_bound_list': [result[2][1].cpu()],
        'gt_list': [result[2][2].cpu()],
        'bound_type': [args.bound_type],
    })
    output.to_pickle(args.output_dir + '/' + args.output_file)
    
    print(f"total time elapsed: {time.time() - tick}")
'''
ex)
python real_data_torch.py --output_dir=data2/output --output_file=output2.pd --bound_type=2 --y_column=column_name --input_file1=input1.csv --input_file2=input2.csv
python real_data_torch.py --output_dir=data2/output --output_file=output1.pd --bound_type=2 --y_column=order --input_file1=data2/separate_item_data/item5035.csv --input_file2=data2/separate_item_data/item7789.csv
'''
