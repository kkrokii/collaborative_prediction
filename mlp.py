from real_data_torch import pairwise_comparison, DataGenerator, split_X_y_and_tensorize
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
        f"{f}" : pd.read_csv(os.path.join(input_dir, f)) 
        for f in os.listdir(input_dir) if os.path.isfile( os.path.join(input_dir, f) )
        }
    return file_dict

def merge_datasets(df1, df2):
    return pd.concat([df1, df2])

###### todo : given a dict of datasets, find feature dim -> construct an mlp accordingly -> train mlp -> extract penultimate layer representation 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(torch.nn.module):
    def __init__(self, in_features, hidden_features, out_features):
        self.hidden_features = hidden_features
        self.linear1 = torch.nn.Linear(in_features = in_features, out_features = self.hidden_features)
        self.relu = torch.nn.ReLU()
        self.hidden_features = hidden_features
        self.linear2 = torch.nn.Linear(in_features = self.hidden_features, out_features = out_features)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
    def change_linear_layer(self, out_features):
        self.linear2 = torch.nn.Linear(in_features = self.hidden_features, out_features = out_features)

def df_loader(df, y_column, batch_size = 64):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(df.drop(y_column, axis=1).values), 
        torch.from_numpy(df[[y_column]].values) 
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)              # batch_size x in_features, batch_size x 1
    return dataloader

def training(df_dict, y_column, hidden_features, epochs = 10, lr=1e-4, momentum=0.9):
    in_features = list(df_dict.values())[0].shape[1] - 1
    model = MLP(in_features, hidden_features, out_features = 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
    loss_list = []
    tik = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for df_name, df in df_dict.items():
            dataloader = df_loader(df, y_column)
            for features, labels in dataloader:                                                                         # batch_size x in_features, batch_size x 1
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = torch.mean((labels - outputs)**2)
                total_loss += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_list.append(total_loss)
        print(f"{epoch}th epoch, total time elapsed: {time.time()-tik}")
    return model

class GreedyClustering:
    def __init__(self, df_dict, y_column):
        self.df_dict = df_dict
        self.y_column = y_column
    def greedy_one_step(self, n1, n2, p, current_df_name, current_df, candidate_df_dict, y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, bound_type=2, threshold = 0.9):
        accuracy_dict = {}
        result_dict = {}
        for candidate_df_name, candidate_df in candidate_df_dict.items():
            print(f"comparison of {current_df_name} with : {candidate_df_name}")
            correct, optimal_alpha, (rhs_bound_list, lhs_bound_list, gt_list) = pairwise_comparison(n1, n2, p, current_df, candidate_df, y_column, train_ratio, estimation_ratio, test_ratio, bound_type)
            accuracy_dict[candidate_df_name] = correct / len(gt_list)
            result_dict[candidate_df_name] = {'correct': correct, 'total': len(gt_list)}
        print(accuracy_dict)
        if len(accuracy_dict)==0:
            target_df_key = None
            return target_df_key, None
        max_df_key = max(accuracy_dict, key=accuracy_dict.get)
        target_df_key = max_df_key if accuracy_dict[max_df_key] > threshold else None
        result_dict[max_df_key]['accuracy'] = accuracy_dict[max_df_key]
        return target_df_key, result_dict[max_df_key]

    def greedy(self, n1, n2, p, current_df_name, current_df, candidate_df_dict, y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, bound_type=2, threshold = 0.9):
        print(f"greedy on {current_df_name}")
        candidate_df_dict_copy = copy.deepcopy(candidate_df_dict)
        merged_df_key_list = []
        target_df_key, result_dict = self.greedy_one_step(n1, n2, p, current_df_name, current_df, candidate_df_dict_copy, y_column, train_ratio, estimation_ratio, test_ratio, bound_type, threshold)
        result_after_merged = dict() if target_df_key is None else result_dict
        while target_df_key is not None:
            merged_df_key_list.append(target_df_key)
            result_after_merged = result_dict
            current_df = merge_datasets(current_df, candidate_df_dict_copy[target_df_key])
            del candidate_df_dict_copy[target_df_key]
            target_df_key, result_dict = self.greedy_one_step(n1, n2, p, current_df_name, current_df, candidate_df_dict_copy, y_column, train_ratio, estimation_ratio, test_ratio, bound_type, threshold)
        return merged_df_key_list, result_after_merged
        
    def clustering(self, n1, n2, p, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, bound_type=2, threshold=0.9):
        cluster = {k:-1 for k in self.df_dict.keys()}
        result = {k:None for k in self.df_dict.keys()}
        candidate_df_dict = copy.deepcopy(self.df_dict)
        total_clusters = 0
        for df_key, df in df_dict.items():
            if cluster[df_key] != -1:
                continue
            total_clusters += 1
            cluster[df_key] = total_clusters
            del candidate_df_dict[df_key]
            merged_df_key_list, merged_result_dict = self.greedy(n1, n2, p, df_key, df, candidate_df_dict, self.y_column, train_ratio, estimation_ratio, test_ratio, bound_type, threshold)
            
            for k in merged_df_key_list:
                cluster[k] = total_clusters
                result[k] = merged_result_dict
                del candidate_df_dict[k]
            result[df_key] = merged_result_dict
        return cluster, result

class ErrorCalculator(DataGenerator):
    # does not account for noise_level
    def __init__(self, df_dict, cluster, y_column):
        self.df_dict = df_dict
        self.cluster = cluster
        self.y_column = y_column
        self.clustered_df_dict = self.merge_clusters()
    
    def merge_clusters(self):
        total_clusters = max(cluster.values())
        clustered_df_dict = dict()
        for i in range(1, total_clusters+1):
            current_cluster_key_list = []
            for k,v in self.cluster.items():
                if v==i:
                    current_cluster_key_list.append(k)

            clustered_df_dict[i] = self.df_dict[ current_cluster_key_list[0] ]
            for j in range(1, len(current_cluster_key_list)):
                clustered_df_dict[i] = merge_datasets(clustered_df_dict[i], self.df_dict[ current_cluster_key_list[j] ])
        return clustered_df_dict
    
    def model_fitting(self, X, y):
        return torch.linalg.inv(X.transpose(-1,-2) @ X) @ X.transpose(-1,-2) @ y
        
    def out_of_sample_error(self, df, n, batch_size, samples=1000):
        """
        beta_hat: batch_size x p x 1
        """
        beta_hat = self.model_fitting(*split_X_y_and_tensorize(df[:n], self.y_column))
        X0, y0 = self.generate_bootstraps(df = df, samples=batch_size, n=samples)                                    # batch_size x samples x p
        return (torch.sum((y0 - X0 @ beta_hat)**2, dim=-2)/samples).squeeze()                               # (batch_size, )
    
    def sum_of_out_of_sample_error(self, n, batch_size, samples=1000):
        sum_of_oos = 0
        for k,v in self.df_dict.items():
            sum_of_oos += self.out_of_sample_error(v, n, batch_size, samples)
        return sum_of_oos
    
    def sum_of_clustered_out_of_sample_error(self, n, batch_size, samples=1000):
        sum_of_oos = 0
        for k,v in self.clustered_df_dict.items():
            sum_of_oos += self.out_of_sample_error(v, n, batch_size, samples)
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
        '--bound_type',
        type=int,
        default=1
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
        '--threshold',
        type=float,
        default=0.9
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
    
    clusterer = GreedyClustering(df_dict, args.y_column)
    cluster, result = clusterer.clustering(n1, n2, p, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, bound_type=args.bound_type, threshold=args.threshold)
    # eg)   cluster = {'file1' : 1, 'file2' : 1, 'file3' : 2, 'file4': 1, 'file5': 3}
    #       result = {'file1' : {'correct' : 10, 'total' : 100, 'accuracy' : 0.1}, 'file2' : ...}
    error_calculator = ErrorCalculator(df_dict, cluster, args.y_column)
    separate_error_sum = error_calculator.sum_of_out_of_sample_error(n1, batch_size=1, samples=1000)
    clustered_error_sum = error_calculator.sum_of_clustered_out_of_sample_error(n1, batch_size=1, samples=1000)
    print(f"cluster: {cluster} \n\t separte oos error sum: {separate_error_sum}, clustered oos error sum: {clustered_error_sum}")
    
    output = pd.DataFrame.from_dict({
        'n1': [n1], 
        'n2': [n2], 
        'p': [p], 
        'cluster': [cluster],
        'threshold': [args.threshold],
        'bound_type': [args.bound_type],
        'separate_oos_error_sum': [separate_error_sum.cpu()],
        'clustered_oos_error_sum': [clustered_error_sum.cpu()],
        #'clustered_result_by_file': [result],
    })
    output.to_pickle(args.output_dir + '/' + args.output_file)
    
    print(f"total time elapsed: {time.time() - tick}")
'''
ex)
python clustering_torch.py --output_dir=cluster_output/data2 --output_file=output1.pd --bound_type=1 --y_column=order --input_dir=data2/separate_item_data --threshold=0.9
'''
