import numpy as np
import pandas as pd
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def bound_constants(self, n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma = 0.9):
        Xc = torch.cat((X1, X2), dim=-2)
        yc = torch.cat((y1, y2), dim=-2)
        X0 = torch.cat((torch.cat((X1, torch.zeros_like(X1)), dim=-1), torch.cat((torch.zeros_like(X2), X2), dim=-1)), dim=-2)
        
        cov_1_inv = self.inverse_of_covariance(X1)
        cov_2_inv = self.inverse_of_covariance(X2)
        cov_c_inv = self.inverse_of_covariance(Xc)
        cov_0_inv = torch.cat((torch.cat((cov_1_inv, torch.zeros_like(cov_1_inv)), dim=-1), torch.cat((torch.zeros_like(cov_2_inv), cov_2_inv), dim=-1)), dim=-2)
        beta_1_hat = self.coefficient(cov_1_inv, X1, y1)
        beta_2_hat = self.coefficient(cov_2_inv, X2, y2)
        beta_c_hat = self.coefficient(cov_c_inv, Xc, yc)
        self.beta_1_hat = beta_1_hat
        self.beta_2_hat = beta_2_hat
        self.beta_c_hat = beta_c_hat
        
        DTD = torch.linalg.inv( cov_1_inv + cov_2_inv )
        
        sigma_c_2_hat = ( torch.sum((y1 - X1 @ beta_c_hat)**2, dim=-2) + torch.sum((y2 - X2 @ beta_c_hat)**2, dim=-2) ) / (n1+n2-p)
        sigma_c_2_hat = sigma_c_2_hat.squeeze()
        
        C = torch.cat((torch.eye(p), -torch.eye(p)), dim=1).to(device)

        A = X0 @ cov_0_inv @ C.transpose(0,1)
        M = A @ self.inverse_of_covariance(A) @ A.transpose(-1,-2)
        
        Sigma = X0 @ cov_0_inv @ C.transpose(0,1) @ BTB @ C @ cov_0_inv @ X0.transpose(-1,-2)
        norm_Sigma = torch.linalg.matrix_norm(Sigma, ord=2).to(device)   

        e = torch.exp(torch.Tensor([1])).to(device)
        
        const0 = ( torch.sqrt((beta_1_hat-beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat-beta_2_hat)) ).squeeze()
        const1 = torch.sqrt((n1+n2-p)*sigma_c_2_hat)
        const9 = torch.log(torch.Tensor([4/(1+1-gamma)])).to(device)
        const2 = -(n1+n2-p)/8*(1+1-gamma)/torch.sqrt(e*const9)
        const3 = -torch.sqrt(2*torch.Tensor([n1+n2-p])).to(device) * (torch.sqrt(const9) + 1/2/torch.sqrt(const9))
        const4 = torch.sqrt( (beta_1_hat - beta_2_hat).transpose(-1,-2) @ BTB @ (beta_1_hat - beta_2_hat) )
        
        const5 = self.trace(Sigma)
        const6 = 2*torch.sqrt(self.trace(Sigma @ Sigma))
        const7 = 2*norm_Sigma
        rhs_bound_ft_const_list = [const0, const1, const2, const3, const4, const5, const6, const7]
        
        const10 = torch.trace(W1 @ lhs_const1 + W2 @ lhs_const2)
        const11 = (n1+n2-p) * sigma_c_2_hat - ( 2*(beta_1_hat - beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat - beta_2_hat) ).squeeze()
        const13 = torch.Tensor([n1+n2-p]).to(device)
        const12 = -4/torch.sqrt(const13) * ( (beta_1_hat - beta_2_hat).transpose(-1,-2) @ DTD @ (beta_1_hat - beta_2_hat) ).squeeze()
        const14 = 2*torch.sqrt(const13)
        const15 = 4/torch.sqrt(const13)
        lhs_bound_ft_const_list = [const10, const11, const12, const13, const14, const15]

        for i in range(len(rhs_bound_ft_const_list)):
            rhs_bound_ft_const_list[i] = rhs_bound_ft_const_list[i].reshape(-1,1).to(device)
        for i in range(len(lhs_bound_ft_const_list)):
            lhs_bound_ft_const_list[i] = lhs_bound_ft_const_list[i].reshape(-1,1).to(device)
        return rhs_bound_ft_const_list, lhs_bound_ft_const_list
    
    def lhs_bound_ft(self, const_list):
        return const_list[0] * (const_list[1] + const_list[2]*torch.sqrt(self.alpha_list)) / (const_list[3] + const_list[4]*torch.sqrt(self.alpha_list) + 2*self.alpha_list) / (3 + const_list[5]*torch.sqrt(self.alpha_list))
    
    def rhs_bound_ft(self, const_list):
        const = ( n1+n2-p + 2*torch.sqrt((n1+n2-p)*self.alpha_list) + 2*self.alpha_list ).reshape(1,-1)
        const_rhs1 = const_list[1]*torch.sqrt(const)
        const_rhs2 = const_list[0]*const_list[1]
        const_rhs3 = const_list[2] + const_list[3] * torch.sqrt(const) + const
        const_rhs4 = const_list[0] * (-const_list[3] - 2*torch.sqrt(const))
        const_rhs5 = const_list[0]**2
        sigma_bound = (const_rhs1-const_rhs4 + torch.sqrt( (const_rhs1-const_rhs4)**2 -4*const_rhs3*(const_rhs5-const_rhs2) ) ) / 2 / const_rhs3
        sigma_term = sigma_bound * torch.sqrt(const_list[5] + const_list[6]*torch.sqrt(self.alpha_list) + const_list[7]*self.alpha_list)
        norm_term = const_list[4]
        return (sigma_term + norm_term)**2
    
    def get_bound_list(self, n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma = 0.9):
        rhs_bound_const_list, lhs_bound_const_list = self.bound_constants(n1, n2, p, X1, y1, X2, y2, W1, W2, BTB, lhs_const1, lhs_const2, gamma)
        rhs_bound_list = self.rhs_bound_ft(rhs_bound_const_list)
        lhs_bound_list = self.lhs_bound_ft(lhs_bound_const_list)
        return rhs_bound_list, lhs_bound_list



class SimulationDataGenerator:
    def __init__(self, mu_X1, var_X1, mu_X2, var_X2, mu_beta=None, var_beta=None, noise_level=1):
        self.mu_X1 = mu_X1
        self.var_X1 = var_X1
        self.mu_X2 = mu_X2
        self.var_X2 = var_X2
        self.mu_beta = mu_beta
        self.var_beta = var_beta
        self.noise_level = noise_level
    def generate_data(self, batch_size, n, beta=None, distribution_index = 1):
        if distribution_index == 1:
            X = torch.distributions.MultivariateNormal(self.mu_X1, self.var_X1).sample([batch_size, n]).to(device) 
        elif distribution_index == 2:
            X = torch.distributions.MultivariateNormal(self.mu_X2, self.var_X2).sample([batch_size, n]).to(device) 

        if beta is None:
            beta = torch.distributions.MultivariateNormal(self.mu_beta, self.var_beta).sample([batch_size])     
            beta = beta.unsqueeze(-1).to(device)                                                                
        noise = torch.distributions.MultivariateNormal(torch.Tensor([0]), torch.Tensor([[self.noise_level]])).sample([batch_size, n]).to(device)
        y = ( X @ beta + noise ).to(device)
        return X, y, beta
    
    def out_of_sample_error(self, beta_true, beta_hat, batch_size, samples=1000, distribution=1):
        X0, y0, _ = self.generate_data(batch_size=batch_size, n=samples, beta=beta_true, distribution_index = distribution)
        return (torch.sum((y0 - X0 @ beta_hat)**2, dim=-2)/samples).squeeze()        

    def gt_combine(self, beta1_true, beta2_true, beta1_hat, beta2_hat, beta_c_hat, batch_size, samples=1000):
        error1 = self.out_of_sample_error(beta1_true, beta1_hat, batch_size, samples, distribution = 1)
        error2 = self.out_of_sample_error(beta2_true, beta2_hat, batch_size, samples, distribution = 2)
        error_combined_model_1 = self.out_of_sample_error(beta1_true, beta_c_hat, batch_size, samples, distribution = 1)
        error_combined_model_2 = self.out_of_sample_error(beta2_true, beta_c_hat, batch_size, samples, distribution = 2)
        gt = error1+error2 > error_combined_model_1 + error_combined_model_2   
        return gt.to(device)



class Estimator:
    def mean_Z1(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples = 1000):
        X1_list = torch.distributions.MultivariateNormal(mu_X1, var_X1).sample([samples, n1]).to(device)    
        X2_list = torch.distributions.MultivariateNormal(mu_X2, var_X2).sample([samples, n2]).to(device)    
        cov_X1_list = X1_list.transpose(1,2) @ X1_list                                                      
        cov_X2_list = X2_list.transpose(1,2) @ X2_list                                                      
        Z1 = torch.sum( torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X1_list , dim=0 )
        return Z1 / samples
    
    def mean_Z2(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples = 1000):
        X1_list = torch.distributions.MultivariateNormal(mu_X1, var_X1).sample([samples, n1]).to(device)    
        X2_list = torch.distributions.MultivariateNormal(mu_X2, var_X2).sample([samples, n2]).to(device)    
        cov_X1_list = X1_list.transpose(1,2) @ X1_list                                                      
        cov_X2_list = X2_list.transpose(1,2) @ X2_list                                                      
        Z2 = torch.sum( torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X2_list , dim=0 )
        return Z2 / samples

    def mean_Z1TW2Z1(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, W2, samples = 1000):
        X1_list = torch.distributions.MultivariateNormal(mu_X1, var_X1).sample([samples, n1]).to(device)
        X2_list = torch.distributions.MultivariateNormal(mu_X2, var_X2).sample([samples, n2]).to(device)
        cov_X1_list = X1_list.transpose(1,2) @ X1_list
        cov_X2_list = X2_list.transpose(1,2) @ X2_list
        Z1_list = torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X1_list
        Z1TW2Z1 = torch.sum( Z1_list.transpose(1,2) @ W2 @ Z1_list, dim=0 )
        return Z1TW2Z1 / samples
    
    def mean_Z2TW1Z2(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, W1, samples = 1000):
        X1_list = torch.distributions.MultivariateNormal(mu_X1, var_X1).sample([samples, n1]).to(device)
        X2_list = torch.distributions.MultivariateNormal(mu_X2, var_X2).sample([samples, n2]).to(device)
        cov_X1_list = X1_list.transpose(1,2) @ X1_list
        cov_X2_list = X2_list.transpose(1,2) @ X2_list
        Z2_list = torch.linalg.inv( cov_X1_list + cov_X2_list ) @ cov_X2_list
        Z2TW1Z2 = torch.sum( Z2_list.transpose(1,2) @ W1 @ Z2_list, dim=0 )
        return Z2TW1Z2 / samples
        
    def BTB(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, W1, W2, samples = 1000):
        return self.mean_Z1TW2Z1(n1, n2, mu_X1, var_X1, mu_X2, var_X2, W2, samples) + self.mean_Z2TW1Z2(n1, n2, mu_X1, var_X1, mu_X2, var_X2, W1, samples) 

    def mean_cov_inv(self, n, mu_X, var_X, samples = 1000):
        X_list = torch.distributions.MultivariateNormal(mu_X, var_X).sample([samples, n]).to(device)
        XTX_list = X_list.transpose(1,2) @ X_list
        XTX_inv = torch.sum( torch.linalg.inv(XTX_list), dim=0 )
        return XTX_inv / samples    
    
    def mean_cov_inv_combined(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples = 1000):
        X1_list = torch.distributions.MultivariateNormal(mu_X1, var_X1).sample([samples, n1]).to(device)
        X2_list = torch.distributions.MultivariateNormal(mu_X2, var_X2).sample([samples, n2]).to(device)
        X1TX1_list = X1_list.transpose(1,2) @ X1_list
        X2TX2_list = X2_list.transpose(1,2) @ X2_list
        XTX_inv = torch.sum( torch.linalg.inv(X1TX1_list + X2TX2_list), dim=0 )
        return XTX_inv / samples    
    
    def lhs_const1(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples = 1000):
        return self.mean_cov_inv(n1, mu_X1, var_X1, samples) - self.mean_cov_inv_combined(n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples)
    
    def lhs_const2(self, n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples = 1000):
        return self.mean_cov_inv(n2, mu_X2, var_X2, samples) - self.mean_cov_inv_combined(n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples)



def simulation(n1, n2, p, W1, W2, BTB, lhs_const1, lhs_const2, batch_size, data_generator, model, identical_beta = False, beta_perturbation = None):
    X1_batch, y1_batch, beta1_true_batch = data_generator.generate_data(batch_size=batch_size, n=n1, distribution_index = 1)
    if beta_perturbation is not None:
        X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=batch_size, n=n2, beta=beta1_true_batch + beta_perturbation, distribution_index = 2)
    elif identical_beta:
        X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=batch_size, n=n2, beta = beta1_true_batch, distribution_index = 2)
    else:
        X2_batch, y2_batch, beta2_true_batch = data_generator.generate_data(batch_size=batch_size, n=n2, distribution_index = 2)
    
    rhs_bound_list, lhs_bound_list = model.get_bound_list(n1, n2, p, X1_batch, y1_batch, X2_batch, y2_batch, W1, W2, BTB, lhs_const1, lhs_const2, gamma=0.9)
    gt_batch = data_generator.gt_combine(beta1_true_batch, beta2_true_batch, model.beta_1_hat, model.beta_2_hat, model.beta_c_hat, batch_size=batch_size, samples=1000)
    gt_batch = gt_batch.reshape(-1,1)
    
    combine = lhs_bound_list > rhs_bound_list
    correct_list = torch.sum( combine == gt_batch, dim=0 )
    return correct_list



def main(n1, n2, p, mu_X1, var_X1, mu_X2, var_X2, mu_beta, var_beta, beta_perturbation=None, noise_level=1, distinct_distributions_train=500, identical_distributions_train=500, distinct_distributions_test=50, identical_distributions_test=50):
    W1 = var_X1 + mu_X1.view(-1,1) @ mu_X1.view(1,-1)
    W2 = var_X2 + mu_X2.view(-1,1) @ mu_X2.view(1,-1)
    
    estimator = Estimator()
    BTB = estimator.BTB(n1, n2, mu_X1, var_X1, mu_X2, var_X2, W1, W2, samples=1000)
    lhs_const1 = estimator.lhs_const1(n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples=1000)
    lhs_const2 = estimator.lhs_const2(n1, n2, mu_X1, var_X1, mu_X2, var_X2, samples=1000)
    
    min_alpha, max_alpha = (2,10)
    alpha_list = torch.arange(min_alpha, max_alpha, 0.01)
    model = Model(alpha_list)
    
    data_generator = SimulationDataGenerator(mu_X1, var_X1, mu_X2, var_X2, mu_beta, var_beta, noise_level=1)
    
    correct_list = simulation(n1, n2, p, W1, W2, BTB, lhs_const1, lhs_const2, distinct_distributions_train, data_generator, model)
    correct_list += simulation(n1, n2, p, W1, W2, BTB, lhs_const1, lhs_const2, identical_distributions_train, data_generator, model, identical_beta=True)
    
    optimal_alpha = alpha_list[torch.argmax(correct_list)]
    alpha_list = torch.Tensor([optimal_alpha]).to(device)
    model = Model(alpha_list)
    
    correct = simulation(n1, n2, p, W1, W2, BTB, lhs_const1, lhs_const2, distinct_distributions_test, data_generator, model, beta_perturbation)
    correct += simulation(n1, n2, p, W1, W2, BTB, lhs_const1, lhs_const2, identical_distributions_test, data_generator, model, identical_beta=True)
    
    return correct, optimal_alpha



if __name__=='__main__':
    parser = argparse.ArgumentParser()
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
        '--n1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--n2',
        type=int,
        default=1
    )
    parser.add_argument(
        '--p',
        type=int,
        default=1
    )
    args = parser.parse_args()
    
    n1=args.n1
    n2=args.n2
    p=args.p
    beta_perturbation=args.beta_perturbation    # if None, randomly sample two betas
    noise_level=1
    
    mu_X1=torch.zeros((p,)).to(device)
    var_X1=torch.eye(p).to(device)
    mu_X2=torch.ones((p,)).to(device)
    var_X2=torch.eye(p).to(device)
    
    mu_beta=torch.zeros((p,)).to(device)
    var_beta=torch.eye(p).to(device)
    
    distinct_distributions_test=1000
    identical_distributions_test=0
    distinct_distributions_train=500
    identical_distributions_train=500
    
    result = main(
        n1=n1, 
        n2=n2, 
        p=p, 
        mu_X1=mu_X1, 
        var_X1=var_X1, 
        mu_X2=mu_X2,
        var_X2=var_X2,
        mu_beta=mu_beta, 
        var_beta=var_beta, 
        beta_perturbation=beta_perturbation, 
        noise_level=noise_level, 
        distinct_distributions_train=distinct_distributions_train,
        identical_distributions_train=identical_distributions_train,
        distinct_distributions_test=distinct_distributions_test,
        identical_distributions_test=identical_distributions_test,
    )
    output = pd.DataFrame.from_dict({
        'n1': [n1], 
        'n2': [n2], 
        'p': [p], 
        'mu_X1': [mu_X1.cpu()], 
        'var_X1': [var_X1.cpu()], 
        'mu_X2': [mu_X2.cpu()], 
        'var_X2': [var_X2.cpu()], 
        'mu_beta': [mu_beta.cpu()], 
        'var_beta': [var_beta.cpu()], 
        'noise_level': [noise_level], 
        'beta_perturbation_level': [beta_perturbation],
        'distinct_distributions_train':[distinct_distributions_train],
        'identical_distributions_train':[identical_distributions_train],
        'identical_distributions_test':[distinct_distributions_test],
        'identical_distributions_test':[identical_distributions_test],
        'correct': [result[0].cpu()],
        'alpha': [result[1].cpu()],
    })
    output.to_pickle(args.output_file)