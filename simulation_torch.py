import numpy as np
import pandas as pd
import time
import argparse
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import os
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DataGenerator:
    def __init__(self, mu_X, var_X, mu_beta=None, var_beta=None, noise_level=1):
        self.mu_X = mu_X
        self.var_X = var_X
        self.mu_beta = mu_beta
        self.var_beta = var_beta
        self.noise_level = noise_level
    def generate_data(self, batch_size, n, beta=None):
        X = torch.distributions.MultivariateNormal(self.mu_X, self.var_X).sample([batch_size, n]).to(device)    # batch_size x n x p
        if beta is None:
            beta = torch.distributions.MultivariateNormal(self.mu_beta, self.var_beta).sample([batch_size])     # batch_size x p
            beta = beta.unsqueeze(-1).to(device)                                                                # batch_size x p x 1
        noise = torch.distributions.MultivariateNormal(torch.Tensor([0]), torch.Tensor([[self.noise_level]])).sample([batch_size, n]).to(device)   # batch_size x n x 1
        y = ( X @ beta + noise ).to(device)
        return X, y, beta
    
    def out_of_sample_error(self, beta_true, beta_hat, batch_size, samples=1000):
        """
        beta_true: batch_size x p x 1,          beta_hat: batch_size x p x 1
        """
        X0, y0, _ = self.generate_data(batch_size=batch_size, n=samples, beta=beta_true)
        return (torch.sum((y0 - X0 @ beta_hat)**2, dim=-2)/samples).squeeze() - self.noise_level        # (batch_size, )

    def out_of_sample_error_combined(self, beta1_true, beta2_true, beta_c_hat, pi, batch_size, samples=2000):
        from_distr1 = int(round(pi*samples))
        X0_1, y0_1, _ = self.generate_data(batch_size, from_distr1, beta1_true)
        X0_2, y0_2, _ = self.generate_data(batch_size, samples-from_distr1, beta2_true)
        return ( torch.sum((y0_1 - X0_1 @ beta_c_hat)**2, dim=-2) + torch.sum((y0_2 - X0_2 @ beta_c_hat)**2, dim=-2) ).squeeze()/samples  - self.noise_level      # (batch_size, )

    def gt_combine(self, beta1_true, beta2_true, beta1_hat, beta2_hat, beta_c_hat, pi, batch_size, samples=1000):
        error1 = self.out_of_sample_error(beta1_true, beta1_hat, batch_size, samples)
        error2 = self.out_of_sample_error(beta2_true, beta2_hat, batch_size, samples)
        error_combined = self.out_of_sample_error_combined(beta1_true, beta2_true, beta_c_hat, pi, batch_size, samples=samples*2)
        gt = error1+error2 > error_combined   # True: should combine;     False: should not combine       # (batch_size, )
        return gt.to(device)



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
    


class Estimator:
    def mean_Z1(self, n1, n2, mu_X, var_X, samples = 1000):
        X1_list = torch.distributions.MultivariateNormal(mu_X, var_X).sample([samples, n1]).to(device)      # samples x n1 x p
        X2_list = torch.distributions.MultivariateNormal(mu_X, var_X).sample([samples, n2]).to(device)      # samples x n2 x p
        cov_X1_list = X1_list.transpose(1,2) @ X1_list                                                     # samples x p x p
        cov_X2_list = X2_list.transpose(1,2) @ X2_list                                                          # samples x p x p
        Z1 = torch.sum( torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X1_list , dim=0 )
        return Z1 / samples

    def mean_Z1TWZ1(self, n1, n2, mu_X, var_X, W, samples = 1000):
        X1_list = torch.distributions.MultivariateNormal(mu_X, var_X).sample([samples, n1]).to(device)
        X2_list = torch.distributions.MultivariateNormal(mu_X, var_X).sample([samples, n2]).to(device)
        cov_X1_list = X1_list.transpose(1,2) @ X1_list
        cov_X2_list = X2_list.transpose(1,2) @ X2_list
        Z1_list = torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X1_list
        Z1TWZ1 = torch.sum( Z1_list.transpose(1,2) @ W @ Z1_list, dim=0 )
        return Z1TWZ1 / samples
        
    def BTB(self, n1, n2, pi, mu_X, var_X, W, samples = 1000):
        mean_Z1 = self.mean_Z1(n1, n2, mu_X, var_X, samples = 1000)
        return self.mean_Z1TWZ1(n1, n2, mu_X, var_X, W, samples) + pi * (W - mean_Z1.transpose(0,1) @ W - W @ mean_Z1)

    def mean_cov_inv(self, n, mu_X, var_X, samples = 1000):
        X_list = torch.distributions.MultivariateNormal(mu_X, var_X).sample([samples, n]).to(device)
        XTX_list = X_list.transpose(1,2) @ X_list
        XTX_inv = torch.sum( torch.linalg.inv(XTX_list), dim=0 )
        return XTX_inv / samples    
    
    def lhs_const(self, n1, n2, mu_X, var_X, samples = 1000):
        return self.mean_cov_inv(n1, mu_X, var_X, samples) + self.mean_cov_inv(n2, mu_X, var_X, samples) - self.mean_cov_inv(n1+n2, mu_X, var_X, samples)
    

    
def main(n1, n2, p, mu_X, var_X, mu_beta, var_beta, beta_perturbation=None, noise_level=1, distinct_distributions_train=500, identical_distributions_train=500, distinct_distributions_test=50, identical_distributions_test=50, bound_type=2):
    tick = time.time()
    print(f"main ft begins, beta_perturbation: {beta_perturbation}")
    #mu_X = torch.zeros(p).to(device)
    W = var_X
    pi = n1/(n1+n2)
    
    estimator = Estimator()
    BTB = estimator.BTB(n1, n2, pi, mu_X, var_X, W, samples=1000)
    lhs_const = estimator.lhs_const(n1, n2, mu_X, var_X, samples=1000)
    actual_lhs_bound = noise_level * torch.trace(W @ lhs_const)
    print(f"\t constants estimated, time: {time.time()-tick}")
    
    data_generator = DataGenerator(mu_X, var_X, mu_beta, var_beta, noise_level=1)
    min_alpha, max_alpha = (2,10)
    alpha_list = torch.arange(min_alpha, max_alpha, 0.01)
    model = Model(alpha_list, bound_type)
    
    X1_batch, y1_batch, beta1_true_batch = data_generator.generate_data(batch_size=distinct_distributions_train, n=n1)
    X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=distinct_distributions_train, n=n2)
    
    rhs_bound_list, lhs_bound_list = model.get_bound_list(n1, n2, p, X1_batch, y1_batch, X2_batch, y2_batch, W, BTB, lhs_const, gamma=0.9)      # batch_size x len(alpha_list), batch_size x len(alpha_list)
    gt_batch = data_generator.gt_combine(beta1_true_batch, beta2_true_batch, model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, pi, batch_size=distinct_distributions_train, samples=1000)     # (batch_size, )
    gt_batch = gt_batch.reshape(-1,1)                      # batch_size x 1 
    
    combine = lhs_bound_list > rhs_bound_list
    correct_list = torch.sum( combine == gt_batch, dim=0 )
    
    X1_batch, y1_batch, beta1_true_batch = data_generator.generate_data(batch_size=identical_distributions_train, n=n1)
    X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=identical_distributions_train, n=n2, beta=beta1_true_batch)
    
    rhs_bound_list, lhs_bound_list = model.get_bound_list(n1, n2, p, X1_batch, y1_batch, X2_batch, y2_batch, W, BTB, lhs_const, gamma=0.9)      # batch_size x len(alpha_list), batch_size x len(alpha_list)
    gt_batch = data_generator.gt_combine(beta1_true_batch, beta2_true_batch, model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, pi, batch_size=identical_distributions_train, samples=1000)     # (batch_size, )
    gt_batch = gt_batch.reshape(-1,1)                      # batch_size x 1 
    
    combine = lhs_bound_list > rhs_bound_list
    correct_list += torch.sum( combine == gt_batch, dim=0 )
    
    optimal_alpha = alpha_list[torch.argmax(correct_list)]
    alpha_list = torch.Tensor([optimal_alpha]).to(device)
    model = Model(alpha_list, bound_type)
    
    print(f"\t hyperparameter tuned, time: {time.time()-tick}, alpha: {optimal_alpha}")
    
    X1_batch, y1_batch, beta1_true_batch = data_generator.generate_data(batch_size = distinct_distributions_test, n=n1)
    if beta_perturbation is not None:
        X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=distinct_distributions_test, n=n2, beta=beta1_true_batch + beta_perturbation)
    else:
        X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=distinct_distributions_test, n=n2)

    rhs_bound_list_distinct, lhs_bound_list_distinct = model.get_bound_list(n1, n2, p, X1_batch, y1_batch, X2_batch, y2_batch, W, BTB, lhs_const, gamma=0.9)      # batch_size x len(alpha_list), batch_size x len(alpha_list)
    gt_list_distinct = data_generator.gt_combine(beta1_true_batch, beta2_true_batch, model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, pi, batch_size=distinct_distributions_test, samples=1000)     # (batch_size, )
    gt_list_distinct = gt_list_distinct.reshape(-1,1)                      # batch_size x 1 
    
    combine = lhs_bound_list_distinct > rhs_bound_list_distinct
    correct = torch.sum( combine == gt_list_distinct )
    actual_rhs_bound_list_distinct = ( (beta1_true_batch-beta2_true_batch).transpose(-1,-2) @ BTB @ (beta1_true_batch-beta2_true_batch) ).squeeze()
    actual_lhs_bound_list_distinct = actual_lhs_bound * torch.ones(distinct_distributions_test).to(device)
    

    X1_batch, y1_batch, beta1_true_batch = data_generator.generate_data(batch_size = identical_distributions_test, n=n1)
    X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=identical_distributions_test, n=n2, beta=beta1_true_batch)
    
    rhs_bound_list_identical, lhs_bound_list_identical = model.get_bound_list(n1, n2, p, X1_batch, y1_batch, X2_batch, y2_batch, W, BTB, lhs_const, gamma=0.9)      # batch_size x len(alpha_list), batch_size x len(alpha_list)
    gt_list_identical = data_generator.gt_combine(beta1_true_batch, beta2_true_batch, model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, pi, batch_size=identical_distributions_test, samples=1000)     # (batch_size, )
    gt_list_identical = gt_list_identical.reshape(-1,1)                      # batch_size x 1 
    
    combine = lhs_bound_list_identical > rhs_bound_list_identical
    correct += torch.sum( combine == gt_list_identical )
    actual_rhs_bound_list_identical = torch.zeros(identical_distributions_test)
    actual_lhs_bound_list_identical = actual_lhs_bound * torch.ones(identical_distributions_test).to(device)
    
    print(f"\t main ft finished, time: {time.time()-tick}")
    
    return correct, optimal_alpha, \
        (rhs_bound_list_distinct, lhs_bound_list_distinct, gt_list_distinct, actual_rhs_bound_list_distinct, actual_lhs_bound_list_distinct), \
            (rhs_bound_list_identical, lhs_bound_list_identical, gt_list_identical, actual_rhs_bound_list_identical, actual_lhs_bound_list_identical)
    
    

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
        '--beta_perturbation',
        type=float,
        default=None
    )
    parser.add_argument(
        '--bound_type',
        type=int,
        default=2
    )
    args = parser.parse_args()
    
    n1=50
    n2=50
    p=10
    mu_X=torch.zeros((p,)).to(device)
    var_X=torch.eye(p).to(device)
    mu_beta=torch.zeros((p,)).to(device)
    var_beta=torch.eye(p).to(device)
    noise_level=1
    beta_perturbation=args.beta_perturbation
    distinct_distributions_test=500
    identical_distributions_test=500
    distinct_distributions_train=500
    identical_distributions_train=500
    
    if not os.path.isdir(args.output_dir):
        pathlib.Path.mkdir(pathlib.Path(args.output_dir), exist_ok=True)
        print(f"Creating {args.output_dir}")
    
    tick = time.time()
    for beta_perturbation in np.arange(0.1, 1.01, 0.1):
        beta_perturbation = np.round(beta_perturbation, 1)
        result = main(
            n1=n1, 
            n2=n2, 
            p=p, 
            mu_X=mu_X, 
            var_X=var_X, 
            mu_beta=mu_beta, 
            var_beta=var_beta, 
            beta_perturbation=beta_perturbation, 
            noise_level=noise_level, 
            distinct_distributions_train=distinct_distributions_train,
            identical_distributions_train=identical_distributions_train,
            distinct_distributions_test=distinct_distributions_test,
            identical_distributions_test=identical_distributions_test,
            bound_type=args.bound_type
        )
        
        output = pd.DataFrame.from_dict({
            'n1': [n1], 
            'n2': [n2], 
            'p': [p], 
            'mu_X': [mu_X.cpu()], 
            'var_X': [var_X.cpu()], 
            'mu_beta': [mu_beta.cpu()], 
            'var_beta': [var_beta.cpu()], 
            'noise_level': [noise_level], 
            'correct': [result[0].cpu()],
            'alpha': [result[1].cpu()],
            'rhs_bound_list_for_distinct_distr': [result[2][0].cpu()],
            'lhs_bound_list_for_distinct_distr': [result[2][1].cpu()],
            'gt_list_for_distinct_distr': [result[2][2].cpu()],
            'actual_rhs_bound_list_distinct': [result[2][3].cpu()],
            'actual_lhs_bound_list_distinct': [result[2][4].cpu()],
            'rhs_bound_list_for_identical_distr': [result[3][0].cpu()],
            'lhs_bound_list_for_identical_distr': [result[3][1].cpu()],
            'gt_list_for_identical_distr': [result[3][2].cpu()],
            'actual_rhs_bound_list_identical': [result[3][3].cpu()],
            'actual_lhs_bound_list_identical': [result[3][4].cpu()],
            'beta_setting': ['random sampling' if beta_perturbation is None else 'uniform perturbation'],
            'beta_perturbation_level': [beta_perturbation],
            'distinct_distributions_train':[distinct_distributions_train],
            'identical_distributions_train':[identical_distributions_train],
            'identical_distributions_test':[distinct_distributions_test],
            'identical_distributions_test':[identical_distributions_test],
            'bound_type': [args.bound_type],
        })
        output.to_pickle(args.output_dir + '/' + args.output_file + f'_{beta_perturbation}.pd')
    print(f"total time elapsed: {time.time() - tick}")
'''
ex)
python simulation_torch.py --output_dir=output_with_errors_torch/50,50,10,500,500 --output_file=output2.pd --beta_perturbation=1
'''
