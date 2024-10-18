from new_real_data_torch import pairwise_comparison, split_X_y_and_tensorize
import numpy as np
import pandas as pd
import time
import argparse
import os
import copy

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import pathlib

def read_files(input_dir):
    file_dict = {
        f"{f}" : pd.read_csv(os.path.join(input_dir, f)).astype(np.float32)
        for f in os.listdir(input_dir) if os.path.isfile( os.path.join(input_dir, f) )
        }
    return file_dict

def merge_datasets(df1, df2):
    return pd.concat([df1, df2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, mlp_layers = 1):
        super().__init__()
        self.hidden_features = hidden_features
        if mlp_layers == 1:
            self.main = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features, out_features = self.hidden_features),
            )
        elif mlp_layers == 2:
            self.main = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features, out_features = self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = self.hidden_features, out_features = self.hidden_features),
            )
        elif mlp_layers == 3:
            self.main = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features, out_features = self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = self.hidden_features, out_features = self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = self.hidden_features, out_features = self.hidden_features),
            )
        else:
            print(f"mlp with {mlp_layers} layers is not supported")
            exit()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(in_features = self.hidden_features, out_features = out_features)
    def forward(self, x):
        return self.linear(self.relu(self.main(x)))
    def change_linear_layer(self, out_features):
        self.linear = torch.nn.Linear(in_features = self.hidden_features, out_features = out_features)
        for layer in self.main:
            try:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
            except:
                print('no weight or bias to freeze in this layer')
                pass
    def get_representation(self, x):
        return self.main(x)

def df_loader(df, y_column, batch_size = 64):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(df.drop(y_column, axis=1).values.astype(np.float32)).type(torch.FloatTensor), 
        torch.from_numpy(df[[y_column]].values.astype(np.float32)).type(torch.FloatTensor) 
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)              # batch_size x in_features, batch_size x 1
    return dataloader

def pretraining(df_dict, y_column, mlp_layers, hidden_features, epochs = 10, lr=1e-4, momentum=0.9):
    in_features = list(df_dict.values())[0].shape[1] - 1
    model = MLP(in_features, hidden_features, out_features = 1, mlp_layers = mlp_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
    tik = time.time()
    total_loss_dict = {df_name: [] for df_name in df_dict.keys()}
    for epoch in range(epochs):
        for df_name, df in df_dict.items():
            total_loss_dict[df_name].append(0)
            dataloader = df_loader(df, y_column)
            for features, labels in dataloader:                                                                         # batch_size x in_features, batch_size x 1
                if torch.isnan(features).any() or torch.isnan(labels).any():
                    print('nan')
                    exit()
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = torch.sum((labels - outputs)**2)
                total_loss_dict[df_name][-1] += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss_dict[df_name][-1] /= len(df)

        print(f"{epoch}th epoch, total time elapsed: {time.time()-tik}, loss: ", end='\t')
        total_loss = 0
        for df_name in df_dict.keys():
            #print(f"\t {df_name}: {total_loss_dict[df_name][-1]}")
            total_loss += total_loss_dict[df_name][-1]
        print(f'{total_loss}')
    return model, total_loss_dict

def finetuning(base_model, cluster_dict, df_dict, y_column, epochs = 10, lr=1e-4, momentum=0.9):
    clustered_df_dict = merge_clusters(cluster_dict, df_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
    tik = time.time()
    ft_model_dict = dict()
    for cluster_number, clustered_df in clustered_df_dict.items():
        current_model = copy.deepcopy(base_model)
        current_model.change_linear_layer(out_features=1)
        current_model = current_model.to(device)
        current_total_loss = 0
        for epoch in range(epochs):
            dataloader = df_loader(clustered_df, y_column)
            for features, labels in dataloader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = current_model(features)
                loss = torch.sum((labels - outputs)**2)
                current_total_loss +=loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            current_total_loss /= len(clustered_df)
            print(f"cluster {cluster_number}, {epoch}th epoch, total time elapsed: {time.time()-tik}, loss: ", end='\t')
            print(f'{current_total_loss}')    
        
        ft_model_dict[cluster_number] = current_model
    return ft_model_dict

def convert_df_to_feature(df_dict, y_column, model):
    new_df_dict = dict()
    for df_name, df in df_dict.items():
        x_feature = []
        y = torch.from_numpy(df[[y_column]].values).to(device)
        
        dataloader = df_loader(df, y_column)
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            x_feature.append(model.get_representation(x_batch))
        x_feature = torch.cat(x_feature)
        new_df_dict[df_name] = pd.DataFrame( torch.cat((y, x_feature), axis=1).detach().cpu().numpy() ).rename(columns={0:y_column})
    return new_df_dict

class GreedyClustering:
    def __init__(self, df_dict, y_column):
        self.df_dict = df_dict
        self.y_column = y_column
    def greedy_one_step(self, n1, n2, p, current_df_name, current_df, candidate_df_dict, y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, threshold = 0.9):
        accuracy_dict = {}
        result_dict = {}
        for candidate_df_name, candidate_df in candidate_df_dict.items():
            print(f"    **comparison of {current_df_name} ({current_df.shape}) with : {candidate_df_name} ({candidate_df.shape})")
            correct, optimal_alpha, (rhs_bound_list, lhs_bound_list, gt_list) = pairwise_comparison(n1, n2, p, current_df, candidate_df, y_column, train_ratio, estimation_ratio, test_ratio)
            accuracy_dict[candidate_df_name] = correct / len(gt_list)
            result_dict[candidate_df_name] = {'correct': correct, 'total': len(gt_list)}
        print(f'    {accuracy_dict}')
        if len(accuracy_dict)==0:
            target_df_key = None
            return target_df_key, None
        max_df_key = max(accuracy_dict, key=accuracy_dict.get)
        target_df_key = max_df_key if accuracy_dict[max_df_key] > threshold else None
        result_dict[max_df_key]['accuracy'] = accuracy_dict[max_df_key]
        return target_df_key, result_dict[max_df_key]

    def greedy(self, n1, n2, p, current_df_name, current_df, candidate_df_dict, y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, threshold = 0.9):
        print(f"  *greedy on {current_df_name}")
        candidate_df_dict_copy = copy.deepcopy(candidate_df_dict)
        merged_df_key_list = []
        target_df_key, result_dict = self.greedy_one_step(n1, n2, p, current_df_name, current_df, candidate_df_dict_copy, y_column, train_ratio, estimation_ratio, test_ratio, threshold)
        result_after_merged = dict() if target_df_key is None else result_dict
        while target_df_key is not None:
            merged_df_key_list.append(target_df_key)
            result_after_merged = result_dict
            current_df = merge_datasets(current_df, candidate_df_dict_copy[target_df_key])
            del candidate_df_dict_copy[target_df_key]
            target_df_key, result_dict = self.greedy_one_step(n1, n2, p, current_df_name, current_df, candidate_df_dict_copy, y_column, train_ratio, estimation_ratio, test_ratio, threshold)
        return merged_df_key_list, result_after_merged
        
    def clustering(self, n1, n2, p, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, threshold=0.9):
        cluster_dict = {k:-1 for k in self.df_dict.keys()}
        result = {k:None for k in self.df_dict.keys()}
        candidate_df_dict = copy.deepcopy(self.df_dict)
        total_clusters = 0
        for df_key, df in self.df_dict.items():
            if cluster_dict[df_key] != -1:
                continue
            total_clusters += 1
            cluster_dict[df_key] = total_clusters
            del candidate_df_dict[df_key]
            merged_df_key_list, merged_result_dict = self.greedy(n1, n2, p, df_key, df, candidate_df_dict, self.y_column, train_ratio, estimation_ratio, test_ratio, threshold)
            
            for k in merged_df_key_list:
                cluster_dict[k] = total_clusters
                result[k] = merged_result_dict
                del candidate_df_dict[k]
            result[df_key] = merged_result_dict
        return cluster_dict, result

def merge_clusters(cluster_dict, df_dict):
    total_clusters = max(cluster_dict.values())
    clustered_df_dict = dict()
    for i in range(1, total_clusters+1):
        current_cluster_key_list = []
        for k,v in cluster_dict.items():
            if v==i:
                current_cluster_key_list.append(k)

        clustered_df_dict[i] = df_dict[ current_cluster_key_list[0] ]
        for j in range(1, len(current_cluster_key_list)):
            clustered_df_dict[i] = merge_datasets(clustered_df_dict[i], df_dict[ current_cluster_key_list[j] ])
    return clustered_df_dict

class ErrorCalculator:
    def __init__(self, df_dict, cluster_dict, y_column):
        self.df_dict = df_dict
        self.cluster_dict = cluster_dict
        self.y_column = y_column
        self.clustered_df_dict = merge_clusters(self.cluster_dict, self.df_dict)
    
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
        
    def model_fitting(self, X, y):
        return torch.linalg.inv(X.transpose(-1,-2) @ X) @ X.transpose(-1,-2) @ y
        
    def out_of_sample_error(self, df, beta_hat, max_out_of_samples=1000):
        """
        beta_hat: total_bootstrapping x p x 1
        """                                                       
        X0, y0 = self.generate_samples(df, 1, max_out_of_samples)         # total_bootstrapping x bootstrap_samples x p
        return torch.mean(torch.sum((y0 - X0 @ beta_hat)**2, dim=-2))     # (1, )
    
    def sum_of_out_of_sample_error_one(self, n, max_out_of_samples=1000):
        sum_of_oos = 0
        for df_name, df in self.df_dict.items():
            in_sample_index_list = np.random.choice(len(df), min(n, len(df)))
            in_sample = df.iloc[in_sample_index_list, :]
            beta_hat = self.model_fitting(*split_X_y_and_tensorize(in_sample, self.y_column))
            sum_of_oos += self.out_of_sample_error(df, beta_hat, max_out_of_samples)
        return sum_of_oos
    
    def sum_of_out_of_sample_error(self, n, max_out_of_samples=1000, bootstrap_size = 1000):
        sum_of_oos = 0
        print('sum of ose \t', end='')
        tick = time.time()
        for i in range(bootstrap_size):
            print(f'{i}({round(time.time()-tick)})', end=' ')
            sum_of_oos += self.sum_of_out_of_sample_error_one(n, max_out_of_samples)
        print()
        return sum_of_oos / bootstrap_size
    
    def sum_of_clustered_out_of_sample_error_one(self, n, max_out_of_samples=1000):
        sum_of_oos = 0
        clustered_beta_hat = dict()
        for clustered_df_name, clusterd_df in self.clustered_df_dict.items():
            in_sample_index_list = np.random.choice(len(clusterd_df), min(n, len(clusterd_df)))                     #replace = False
            in_sample = clusterd_df.iloc[in_sample_index_list, :]
            beta_hat = self.model_fitting(*split_X_y_and_tensorize(in_sample, self.y_column))
            clustered_beta_hat[clustered_df_name] = beta_hat
        for df_name, df in self.df_dict.items():
            sum_of_oos += self.out_of_sample_error(df, clustered_beta_hat[ self.cluster_dict[df_name] ], max_out_of_samples)
        return sum_of_oos
    
    def sum_of_clustered_out_of_sample_error(self, n, max_out_of_samples=1000, bootstrap_size=1000):
        sum_of_oos = 0
        print('sum of clustered ose \t', end='')
        for i in range(bootstrap_size):
            print(f'{i}({round(time.time()-tick)})', end=' ')
            sum_of_oos += self.sum_of_clustered_out_of_sample_error_one(n, max_out_of_samples)
        print()
        return sum_of_oos / bootstrap_size

class NNErrorCalculator(ErrorCalculator):
    def __init__(self, df_dict, cluster_dict, y_column, base_model, ft_model_dict):
        self.df_dict = df_dict
        self.cluster_dict = cluster_dict
        self.y_column = y_column
        self.clustered_df_dict = merge_clusters(self.cluster_dict, self.df_dict)
        self.base_model = base_model
        self.ft_model_dict = ft_model_dict
            
    def out_of_sample_error(self, df, model, bootstrap_samples=1000, total_bootstrapping=1000):
        X0, y0 = self.generate_samples(df, bootstrap_samples, total_bootstrapping)       # total_bootstrapping x bootstrap_samples x p
        output = model(X0)                                                                # total_bootstrapping x bootstrap_samples x out_features(=1)
        return torch.mean(torch.sum((y0 - output)**2, dim=-2))                # (1, )
    
    def sum_of_out_of_sample_error(self, bootstrap_samples=1000):
        sum_of_oos = 0
        for df_name, df in self.df_dict.items():
            sum_of_oos += self.out_of_sample_error(df, self.base_model, bootstrap_samples)
        return sum_of_oos
    
    def sum_of_clustered_out_of_sample_error(self, bootstrap_samples=1000):
        sum_of_oos = 0
        for df_name, df in self.clustered_df_dict.items():
            sum_of_oos += self.out_of_sample_error(df, self.ft_model_dict[df_name], bootstrap_samples)
        return sum_of_oos



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
        '--previous_output_file',
        type=str,
        default=None
    )
    parser.add_argument(
        '--y_column',
        type=str,
        default=None
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default=None
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.9
    )
    parser.add_argument(
        '--pretrained',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--mlp_layers',
        type=int,
        default=1
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Creating {args.output_dir}")
    
    tick = time.time()
    
    n1=50
    n2=50
    df_dict = read_files(args.input_dir)
    p=list(df_dict.values())[0].shape[1]-1
    
    lr = 1e-4
    momentum = 0.9
    hidden_features = p
    epochs = 10
    
    max_out_of_samples = 100
    bootstrap_size = 10
    
    if args.pretrained:
        in_features = list(df_dict.values())[0].shape[1] - 1
        model = MLP(in_features, hidden_features, out_features = 1, mlp_layers = args.mlp_layers)
        model.load_state_dict(torch.load(args.output_dir + '/' + args.output_model, weights_only=True))
        model = model.to(device)
        print("pretrained model loaded")
    else:
        print(f"input dim: {p}, hidden features: {hidden_features}, lr: {lr}, momentum: {momentum}, pretraining epoch: {epochs}")
        model, total_loss_dict = pretraining(df_dict, args.y_column, args.mlp_layers, hidden_features = hidden_features, epochs = epochs, lr=lr, momentum=momentum)
        torch.save(model.state_dict(), args.output_dir + '/' + args.output_model)
        output = pd.DataFrame.from_dict({
            'n1': [n1], 
            'n2': [n2], 
            'p': [p], 
            'threshold': [args.threshold],
            'lr': [lr],
            'momentum': [momentum],
            'epochs': [epochs],
            'loss_list': [total_loss_dict],
            'hidden_features': [hidden_features],
            'time': [time.time() - tick]
            })
        output.to_pickle(args.output_dir + '/' + args.output_file)
        print(f"finished pretraining")
        
    if args.previous_output_file:    
        previous_output_file = pd.read_pickle(args.output_dir + '/' + args.previous_output_file).to_dict('records')[0]
        cluster_dict = previous_output_file['cluster']
        try:
            total_loss_dict = previous_output_file['loss_list']
        except:
            pass
        feature_df_dict = convert_df_to_feature(df_dict, args.y_column, model)
        print('finished loading previous output file')
    else:
        feature_df_dict = convert_df_to_feature(df_dict, args.y_column, model)
        p=list(feature_df_dict.values())[0].shape[1]-1
        clusterer = GreedyClustering(feature_df_dict, args.y_column)
        cluster_dict, result = clusterer.clustering(n1, n2, p, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, threshold=args.threshold)
        # eg)   cluster = {'file1' : 1, 'file2' : 1, 'file3' : 2, 'file4': 1, 'file5': 3}
        #       result = {'file1' : {'correct' : 10, 'total' : 100, 'accuracy' : 0.1}, 'file2' : ...}
        output_dict = {
            'n1': [n1], 
            'n2': [n2], 
            'p': [p], 
            'cluster': [cluster_dict],
            'threshold': [args.threshold],
            'lr': [lr],
            'momentum': [momentum],
            'epochs': [epochs],
            'hidden_features': [hidden_features],
            'time': [time.time() - tick]
        }
        output = pd.DataFrame.from_dict(output_dict)
        output.to_pickle(args.output_dir + '/' + args.output_file)
        print('finished clustering')
        
    error_calculator = ErrorCalculator(feature_df_dict, cluster_dict, args.y_column)
    separate_feature_error_sum = error_calculator.sum_of_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    clustered_feature_error_sum = error_calculator.sum_of_clustered_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    print(f"cluster: {cluster_dict} \n\t (feature) separte oos error sum: {separate_feature_error_sum}, clustered oos error sum: {clustered_feature_error_sum}")
    
    # ft_model_dict = finetuning(model, cluster_dict, df_dict, args.y_column, epochs, lr, momentum)
    # nn_error_calculator = NNErrorCalculator(df_dict, cluster_dict, args.y_column, model, ft_model_dict)
    # separate_raw_error_sum = nn_error_calculator.sum_of_out_of_sample_error(bootstrap_samples=1000)
    # clustered_raw_error_sum = nn_error_calculator.sum_of_clustered_out_of_sample_error(bootstrap_samples=1000)
    # print(f"\t (original data on nn) separte oos error sum: {separate_raw_error_sum}, clustered oos error sum: {clustered_raw_error_sum}")
    
    error_calculator = ErrorCalculator(df_dict, cluster_dict, args.y_column)
    separate_original_error_sum = error_calculator.sum_of_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    clustered_original_error_sum = error_calculator.sum_of_clustered_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    print(f"\t (original data) separte oos error sum: {separate_original_error_sum}, clustered oos error sum: {clustered_original_error_sum}")
    
    output_dict = {
        'n1': [n1], 
        'n2': [n2], 
        'p': [p], 
        'cluster': [cluster_dict],
        'threshold': [args.threshold],
        'separate_feature_oos_error_sum': [separate_feature_error_sum.cpu().detach().numpy()],
        'clustered_feature_oos_error_sum': [clustered_feature_error_sum.cpu().detach().numpy()],
        # 'separate_raw_oos_error_sum': [separate_raw_error_sum.cpu().detach().numpy()],
        # 'clustered_raw_oos_error_sum': [clustered_raw_error_sum.cpu().detach().numpy()],
        'separate_original_oos_error_sum': [separate_original_error_sum.cpu().detach().numpy()],
        'clustered_original_oos_error_sum': [clustered_original_error_sum.cpu().detach().numpy()],
        'lr': [lr],
        'momentum': [momentum],
        'epochs': [epochs],
        'hidden_features': [hidden_features],
        'time': [time.time() - tick]
        #'clustered_result_by_file': [result],
    }
    output = pd.DataFrame.from_dict(output_dict)
    output.to_pickle(args.output_dir + '/' + args.output_file)
    try:
        print(f'loss history: {total_loss_dict}')
        output_dict['loss_list'] = [total_loss_dict]
    except:
        pass
    output = pd.DataFrame.from_dict(output_dict)
    output.to_pickle(args.output_dir + '/' + args.output_file)
        
    print(f"total time elapsed: {time.time() - tick}")
'''
ex)
python new_mlp.py --output_dir=new_output_for_paper/mlp --output_file=output1.pd --y_column=Sales --input_dir=mlp_data1/separate_data --threshold=0.9 --output_model=model1.pth --pretrained=false --mlp_layers=1
python new_mlp.py --output_dir=new_output_for_paper/mlp --output_file=output2.pd --y_column=Weekly_Sales --input_dir=mlp_data2/separate_data/1 --threshold=0.9 --output_model=model2.pth --pretrained=false --mlp_layers=1
python new_mlp.py --output_dir=new_output_for_paper/mlp --output_file=output3.pd --y_column=unit_sales --input_dir=mlp_data3/separate_data/1 --threshold=0.9 --output_model=model3.pth --pretrained=false --mlp_layers=1
'''
