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
    def __init__(self, alpha_list):
        self.alpha_list = torch.Tensor(alpha_list).to(device)
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
    
    def new_bound_constants(self, n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma = 0.9):
        """
        Xi : total_trials x ni x p,   yi : total_trials x ni x 1
        """
        Xc = torch.cat((X1, X2), dim=-2)                                                        # total_trials x (n1+n2) x p
        yc = torch.cat((y1, y2), dim=-2)                                                        # total_trials x (n1+n2) x 1
        X0 = torch.cat((torch.cat((X1, torch.zeros_like(X1)), dim=-1), torch.cat((torch.zeros_like(X2), X2), dim=-1)), dim=-2)   # total_trials x (n1+n2) x 2p
        y0 = yc
        
        cov_1_inv = self.inverse_of_covariance(X1)                                              # total_trials x p x p
        cov_2_inv = self.inverse_of_covariance(X2)
        cov_c_inv = self.inverse_of_covariance(Xc)  
        cov_0_inv = torch.cat((torch.cat((cov_1_inv, torch.zeros_like(cov_1_inv)), dim=-1), torch.cat((torch.zeros_like(cov_2_inv), cov_2_inv), dim=-1)), dim=-2)   # total_trials x 2p x 2p
        beta_1_hat = self.coefficient(cov_1_inv, X1, y1)                                        # total_trials x p x 1
        beta_2_hat = self.coefficient(cov_2_inv, X2, y2)
        beta_c_hat = self.coefficient(cov_c_inv, Xc, yc)
        self.beta_1_hat = beta_1_hat
        self.beta_2_hat = beta_2_hat
        self.beta_c_hat = beta_c_hat
        
        DTD = torch.linalg.inv( cov_1_inv + cov_2_inv )
        
        sigma_c_2_hat = ( torch.sum((y1 - X1 @ beta_c_hat)**2, dim=-2) + torch.sum((y2 - X2 @ beta_c_hat)**2, dim=-2) ) / (n1+n2-p)
        sigma_c_2_hat = sigma_c_2_hat.squeeze()                                                 # (total_trials, )
        
        C = torch.cat((torch.eye(p), -torch.eye(p)), dim=1).to(device)

        A = X0 @ cov_0_inv @ C.transpose(0,1)                                                   # total_trials x (n1+n2) x p
        M = A @ self.inverse_of_covariance(A) @ A.transpose(-1,-2)                              # total_trials x (n1+n2) x (n1+n2)
        
        Sigma = X0 @ cov_0_inv @ C.transpose(0,1) @ BTB @ C @ cov_0_inv @ X0.transpose(-1,-2)   # total_trials x (n1+n2) x (n1+n2)
        norm_Sigma = torch.linalg.matrix_norm(Sigma, ord=2).to(device)                          # (total_trials, )

        e = torch.exp(torch.Tensor([1])).to(device)
        const1 = (n1+n2-p)*sigma_c_2_hat + 2 * (y0.transpose(-1,-2) @ M @ y0).squeeze()
        const2 = self.trace(Sigma)
        const3 = 2*torch.sqrt(self.trace(Sigma @ Sigma))
        const4 = 2*norm_Sigma
        const5 = (1-gamma)**(2/(n1+n2-p)) * (n1+n2-p)/2/e -2*(n1+n2-p)
        const6 = ( -4*torch.sqrt(torch.Tensor([n1+n2-p])) ).to(device)
        const7 = torch.sqrt( (beta_1_hat - beta_2_hat).transpose(-1,-2) @ BTB @ (beta_1_hat - beta_2_hat) )
        rhs_bound_ft_const_list = [const1, const2, const3, const4, const5, const6, const7]
        
        const28 = torch.Tensor([n1+n2-p]).to(device)
        const20 = ( torch.sqrt((beta_1_hat-beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat-beta_2_hat)) ).squeeze()
        const21 = torch.sqrt((n1+n2-p)*sigma_c_2_hat)
        const29 = torch.log(torch.Tensor([4/(1+1-gamma)])).to(device)
        const22 = -(n1+n2-p)/8*(1+1-gamma)/torch.sqrt(e*const29)
        const23 = -torch.sqrt(2*torch.Tensor([n1+n2-p])).to(device) * (torch.sqrt(const29) + 1/2/torch.sqrt(const29))
        const24 = torch.sqrt( (beta_1_hat - beta_2_hat).transpose(-1,-2) @ BTB @ (beta_1_hat - beta_2_hat) )
        
        const25 = self.trace(Sigma)
        const26 = 2*torch.sqrt(self.trace(Sigma @ Sigma))
        const27 = 2*norm_Sigma
        rhs_bound_ft_const_list = [const20, const21, const22, const23, const24, const25, const26, const27, const28]
        
        const11 = torch.trace(W1 @ lhs_const1 + W2 @ lhs_const2)
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
        return rhs_bound_ft_const_list, lhs_bound_ft_const_list                     # constants are either total_trials x 1 or 1 x 1
    
    def lhs_bound_ft(self, const_list):
        return const_list[0] * (const_list[1] + const_list[2]*torch.sqrt(self.alpha_list)) / (const_list[3] + const_list[4]*torch.sqrt(self.alpha_list) + 2*self.alpha_list) / (3 + const_list[5]*torch.sqrt(self.alpha_list))
    
    def rhs_bound_ft00(self, const_list, print_results = False):
        const = ( const_list[8] + 2*torch.sqrt((const_list[8])*self.alpha_list) + 2*self.alpha_list ).reshape(1,-1)
        const_rhs1 = const_list[1]*torch.sqrt(const)
        const_rhs2 = const_list[0]*const_list[1]
        const_rhs3 = const_list[2] + const_list[3] * torch.sqrt(const) + const
        const_rhs4 = const_list[0] * (-const_list[3] - 2*torch.sqrt(const))
        const_rhs5 = const_list[0]**2
        sigma_bound = (const_rhs1-const_rhs4 + torch.sqrt( (const_rhs1-const_rhs4)**2 -4*const_rhs3*(const_rhs5-const_rhs2) ) ) / 2 / const_rhs3
        sigma_term = sigma_bound * torch.sqrt(const_list[5] + const_list[6]*torch.sqrt(self.alpha_list) + const_list[7]*self.alpha_list)
        norm_term = const_list[4]
        if print_results:
            print(f"\t sigma_bound: {sigma_bound[0].reshape(-1)}, sigma_term: {sigma_term[0].reshape(-1)}, norm_term: {norm_term[0].reshape(-1)}")
        return (sigma_term + norm_term)**2
        
    def get_bound_list0(self, n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma = 0.9, print_results = False):
        rhs_bound_const_list, lhs_bound_const_list = self.new_bound_constants(n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma)
        rhs_bound_list = self.rhs_bound_ft00(rhs_bound_const_list, print_results)
        lhs_bound_list = self.lhs_bound_ft(lhs_bound_const_list)
        return rhs_bound_list, lhs_bound_list                   # total_trials x len(alpha_list), total_trials x len(alpha_list)



class DataGenerator:
    def __init__(self, y_column):
        self.y_column = y_column
        
    def generate_samples(self, df, n, samples = 1000, replace = True):
        p = df.shape[-1]-1                                                          # exclude y_column
        X_samples = torch.zeros(size=(samples, n, p)).to(device)
        y_samples = torch.zeros(size=(samples, n, 1)).to(device)
        for i in range(samples):
            if replace:
                sampled_df = df.sample(n, replace=replace)                # df with y_column included
            else:
                sampled_df = df.sample(max(n, len(df)), replace=replace)
            X, y = split_X_y_and_tensorize(df=sampled_df, y_column=self.y_column)   # 1 x n x p, 1 x n x 1 
            X_samples[i] = X
            y_samples[i] = y
        return X_samples, y_samples                                                 # samples x n x p
    
    def generate_samples_and_indices(self, df, n, samples = 1000, replace = True):
        p = df.shape[-1]-1                                                          # exclude y_column
        X_samples = torch.zeros(size=(samples, n, p)).to(device)
        y_samples = torch.zeros(size=(samples, n, 1)).to(device)
        index_list = np.zeros(size=(samples, n)).to(device)
        for i in range(samples):
            indices = np.random.choice(len(df), n, replace=False)
            index_list[i] = indices
            sampled_df = df.iloc[indices,:]                                         # df with y_column included
            X, y = split_X_y_and_tensorize(df=sampled_df, y_column=self.y_column)   # 1 x n x p, 1 x n x 1 
            X_samples[i] = X
            y_samples[i] = y
        return X_samples, y_samples, index_list                                     # samples x n x p


class Estimator(DataGenerator):
    def __init__(self, estimation_data1, estimation_data2, y_column):
        self.data1 = estimation_data1       # estimation_data : df;     n x p
        self.data2 = estimation_data2
        self.y_column = y_column

    def mean_Z1TW2Z1(self, n1, n2, W2, samples = 1000):
        X1_list, _ = self.generate_samples(self.data1, n1, samples)
        X2_list, _ = self.generate_samples(self.data2, n2, samples)
        cov_X1_list = X1_list.transpose(1,2) @ X1_list
        cov_X2_list = X2_list.transpose(1,2) @ X2_list
        Z1_list = torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X1_list
        Z1TW2Z1 = torch.sum( Z1_list.transpose(1,2) @ W2 @ Z1_list, dim=0 )
        return Z1TW2Z1 / samples
    
    def mean_Z2TW1Z2(self, n1, n2, W1, samples = 1000):
        X1_list, _ = self.generate_samples(self.data1, n1, samples)
        X2_list, _ = self.generate_samples(self.data2, n2, samples)
        cov_X1_list = X1_list.transpose(1,2) @ X1_list
        cov_X2_list = X2_list.transpose(1,2) @ X2_list
        Z2_list = torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X2_list
        Z2TW1Z2 = torch.sum( Z2_list.transpose(1,2) @ W1 @ Z2_list, dim=0 )
        return Z2TW1Z2 / samples
    
    def BTB(self, n1, n2, W1, W2, samples = 1000):
        return self.mean_Z1TW2Z1(n1, n2, W2, samples) + self.mean_Z2TW1Z2(n1, n2, W1, samples) 

    def mean_cov_inv(self, n, samples = 1000, distribution = 1):
        df = self.data1 if distribution==1 else self.data2
        X_list, _ = self.generate_samples(df, n, samples)
        XTX_list = X_list.transpose(1,2) @ X_list
        XTX_inv = torch.sum( torch.linalg.inv(XTX_list), dim=0 )
        return XTX_inv / samples    
    
    def mean_cov(self, n, samples = 1000, distribution = 1):
        df = self.data1 if distribution==1 else self.data2
        X_list, _ = self.generate_samples(df, n, samples)
        XTX_list = X_list.transpose(1,2) @ X_list
        XTX = torch.sum( XTX_list, dim=0 )
        return XTX / samples    
    
    def mean_cov_inv_combined(self, n1, n2, samples = 1000):
        X1_list, _ = self.generate_samples(self.data1, n1, samples)
        X2_list, _ = self.generate_samples(self.data2, n2, samples)
        X1TX1_list = X1_list.transpose(1,2) @ X1_list
        X2TX2_list = X2_list.transpose(1,2) @ X2_list
        XTX_inv = torch.sum( torch.linalg.inv(X1TX1_list + X2TX2_list), dim=0 )
        return XTX_inv / samples    
    
    def lhs_const1(self, n1, n2, samples = 1000):
        return self.mean_cov_inv(n1, samples, distribution=1) - self.mean_cov_inv_combined(n1, n2, samples)
    
    def lhs_const2(self, n1, n2, samples = 1000):
        return self.mean_cov_inv(n2, samples, distribution=2) - self.mean_cov_inv_combined(n1, n2, samples)



class GTGenerator(DataGenerator):
    def __init__(self, df1, df2, y_column):
        self.data1 = df1
        self.data2 = df2
        self.y_column = y_column
        
    def out_of_sample_error(self, beta_hat, X, y, distribution=1):
        """
        beta_hat: total_trials x p x 1
        """
        df = self.data1 if distribution==1 else self.data2
        X0, y0 = self.generate_samples(df = df, samples=1, n=len(df), replace=False)        # 1 x len(df) x p
        return (torch.sum((y0 - X0 @ beta_hat)**2, dim=-2) - torch.sum((y - X @ beta_hat)**2, dim=-2)).squeeze() / (len(df) - len(X))                      # (total_trials, )

    def gt_combine(self, beta1_hat, beta2_hat, beta_c_hat, X1, y1, X2, y2):
        error1 = self.out_of_sample_error(beta1_hat, X1, y1, distribution=1)
        error2 = self.out_of_sample_error(beta2_hat, X2, y2, distribution=2)
        error_combined_model_1 = self.out_of_sample_error(beta_c_hat, X1, y1, distribution = 1)
        error_combined_model_2 = self.out_of_sample_error(beta_c_hat, X2, y2, distribution = 2)
        gt = error1+error2 > error_combined_model_1 + error_combined_model_2   # True: should combine;     False: should not combine       # (total_trials, )
        return gt.to(device)



def pairwise_comparison(n1, n2, p, df1, df2, y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2):
    tick = time.time()
    
    data_splitter = DataSplitter()
    train_data1, estimation_data1, test_data1 = data_splitter.split_train_estimation_test(df1, train = train_ratio, estimation = estimation_ratio, test = test_ratio)
    train_data2, estimation_data2, test_data2 = data_splitter.split_train_estimation_test(df2, train = train_ratio, estimation = estimation_ratio, test = test_ratio)
    # need to consider these as distr;      further bootstrap from each distr
    
    estimator = Estimator(estimation_data1, estimation_data2, y_column)
    W1 = estimator.mean_cov(n=1, samples = 1000, distribution=1)
    W2 = estimator.mean_cov(n=1, samples = 1000, distribution=2)
    BTB = estimator.BTB(n1, n2, W1, W2, samples = 1000)
    lhs_const1 = estimator.lhs_const1(n1, n2, samples = 1000)
    lhs_const2 = estimator.lhs_const2(n1, n2, samples = 1000)
    print(f"\t constants estimated, time: {time.time()-tick}")
    
    data_generator = DataGenerator(y_column)
    total_trials=1000
    
    min_alpha, max_alpha = (2,10)
    alpha_list = torch.arange(min_alpha, max_alpha, 0.01)
    model = Model(alpha_list)
    
    X1, y1 = data_generator.generate_samples(df=train_data1, samples=total_trials, n=n1)    # total_trials x n1 x p, total_trials x n1 x 1
    X2, y2 = data_generator.generate_samples(df=train_data2, samples=total_trials, n=n2)    # total_trials x n1 x p, total_trials x n1 x 1
    
    gt_generator = GTGenerator(train_data1, train_data2, y_column)
    rhs_bound_list, lhs_bound_list = model.get_bound_list0(n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma = 0.9)     # total_trials x len(alpha_list), total_trials x len(alpha_list)
    gt_list = gt_generator.gt_combine(model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, X1, y1, X2, y2)                       # (total_trials, )
    gt_list = gt_list.reshape(-1,1)                                                         # total_trials x 1 
    
    combine = lhs_bound_list > rhs_bound_list
    correct_list = torch.sum( combine == gt_list, dim=0 )
    
    optimal_alpha = alpha_list[torch.argmax(correct_list)]
    alpha_list = torch.Tensor([optimal_alpha]).to(device)
    model = Model(alpha_list)
    
    print(f"\t hyperparameter tuned, time: {time.time()-tick}, alpha: {optimal_alpha}")
    
    X1, y1 = data_generator.generate_samples(df=test_data1, samples=total_trials, n=n1)     # total_trials x n1 x p, total_trials x n1 x 1
    X2, y2 = data_generator.generate_samples(df=test_data2, samples=total_trials, n=n2)     # total_trials x n1 x p, total_trials x n1 x 1
    
    gt_generator = GTGenerator(test_data1, test_data2, y_column)
    rhs_bound_list, lhs_bound_list = model.get_bound_list0(n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma=0.9)      # total_trials x len(alpha_list), total_trials x len(alpha_list)
    gt_list = gt_generator.gt_combine(model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, X1, y1, X2, y2)     # (total_trials, )
    gt_list = gt_list.reshape(-1,1)                      # total_trials x 1 
    
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
    result = pairwise_comparison(n1, n2, p, df1, df2, args.y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2)
    
    output = pd.DataFrame.from_dict({
        'n1': [n1], 
        'n2': [n2], 
        'p': [p], 
        'correct': [result[0].cpu()],
        'alpha': [result[1].cpu()],
        'rhs_bound_list': [result[2][0].cpu()],
        'lhs_bound_list': [result[2][1].cpu()],
        'gt_list': [result[2][2].cpu()],
    })
    output.to_pickle(args.output_dir + '/' + args.output_file)
    
    print(f"total time elapsed: {time.time() - tick}")
'''
ex)
python new_real_data_torch.py --output_dir=new_output_for_paper --output_file=output2.pd --y_column=order --input_file1=data2/separate_item_data/item5035.csv --input_file2=data2/separate_item_data/item7789.csv
'''
