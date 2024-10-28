from real_data import pairwise_comparison, split_X_y_and_tensorize
import numpy as np
import pandas as pd
import time
import argparse
import os
import copy

import torch

def read_files(input_dir):
    file_dict = {f"{f}" : pd.read_csv(os.path.join(input_dir, f)).astype(np.float32) for f in os.listdir(input_dir) if os.path.isfile( os.path.join(input_dir, f) )}
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
                pass
    def get_representation(self, x):
        return self.main(x)

def df_loader(df, y_column, batch_size = 64):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(df.drop(y_column, axis=1).values.astype(np.float32)).type(torch.FloatTensor), 
        torch.from_numpy(df[[y_column]].values.astype(np.float32)).type(torch.FloatTensor) 
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True) 
    return dataloader

def pretraining(df_dict, y_column, mlp_layers, hidden_features, epochs = 10, lr=1e-4, momentum=0.9):
    in_features = list(df_dict.values())[0].shape[1] - 1
    model = MLP(in_features, hidden_features, out_features = 1, mlp_layers = mlp_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
    total_loss_dict = {df_name: [] for df_name in df_dict.keys()}
    for epoch in range(epochs):
        for df_name, df in df_dict.items():
            total_loss_dict[df_name].append(0)
            dataloader = df_loader(df, y_column)
            for features, labels in dataloader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = torch.sum((labels - outputs)**2)
                total_loss_dict[df_name][-1] += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss_dict[df_name][-1] /= len(df)

        total_loss = 0
        for df_name in df_dict.keys():
            total_loss += total_loss_dict[df_name][-1]
    return model, total_loss_dict

def finetuning(base_model, cluster_dict, df_dict, y_column, epochs = 10, lr=1e-4, momentum=0.9):
    clustered_df_dict = merge_clusters(cluster_dict, df_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
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
            correct, optimal_alpha, (rhs_bound_list, lhs_bound_list, gt_list) = pairwise_comparison(n1, n2, p, current_df, candidate_df, y_column, train_ratio, estimation_ratio, test_ratio)
            accuracy_dict[candidate_df_name] = correct / len(gt_list)
            result_dict[candidate_df_name] = {'correct': correct, 'total': len(gt_list)}
        if len(accuracy_dict)==0:
            target_df_key = None
            return target_df_key, None
        max_df_key = max(accuracy_dict, key=accuracy_dict.get)
        target_df_key = max_df_key if accuracy_dict[max_df_key] > threshold else None
        result_dict[max_df_key]['accuracy'] = accuracy_dict[max_df_key]
        return target_df_key, result_dict[max_df_key]

    def greedy(self, n1, n2, p, current_df_name, current_df, candidate_df_dict, y_column, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, threshold = 0.9):
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
        p = df.shape[-1]-1                                                
        X_samples = torch.zeros(size=(samples, n, p)).to(device)
        y_samples = torch.zeros(size=(samples, n, 1)).to(device)
        for i in range(samples):
            if replace:
                sampled_df = df.sample(n, replace=replace)                
            else:
                sampled_df = df.sample(max(n, len(df)), replace=replace)
            X, y = split_X_y_and_tensorize(df=sampled_df, y_column=self.y_column)
            X_samples[i] = X
            y_samples[i] = y
        return X_samples, y_samples                                      
        
    def model_fitting(self, X, y):
        return torch.linalg.inv(X.transpose(-1,-2) @ X) @ X.transpose(-1,-2) @ y
        
    def out_of_sample_error(self, df, beta_hat, max_out_of_samples=1000):                           
        X0, y0 = self.generate_samples(df, 1, max_out_of_samples)        
        return torch.mean(torch.sum((y0 - X0 @ beta_hat)**2, dim=-2))    
    
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
        tick = time.time()
        for i in range(bootstrap_size):
            sum_of_oos += self.sum_of_out_of_sample_error_one(n, max_out_of_samples)
        return sum_of_oos / bootstrap_size
    
    def sum_of_clustered_out_of_sample_error_one(self, n, max_out_of_samples=1000):
        sum_of_oos = 0
        clustered_beta_hat = dict()
        for clustered_df_name, clusterd_df in self.clustered_df_dict.items():
            in_sample_index_list = np.random.choice(len(clusterd_df), min(n, len(clusterd_df)))                     
            in_sample = clusterd_df.iloc[in_sample_index_list, :]
            beta_hat = self.model_fitting(*split_X_y_and_tensorize(in_sample, self.y_column))
            clustered_beta_hat[clustered_df_name] = beta_hat
        for df_name, df in self.df_dict.items():
            sum_of_oos += self.out_of_sample_error(df, clustered_beta_hat[ self.cluster_dict[df_name] ], max_out_of_samples)
        return sum_of_oos
    
    def sum_of_clustered_out_of_sample_error(self, n, max_out_of_samples=1000, bootstrap_size=1000):
        sum_of_oos = 0
        for i in range(bootstrap_size):
            sum_of_oos += self.sum_of_clustered_out_of_sample_error_one(n, max_out_of_samples)
        return sum_of_oos / bootstrap_size



if __name__=='__main__':
    parser = argparse.ArgumentParser()
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
        '--mlp_layers',
        type=int,
        default=1
    )
    args = parser.parse_args()
    
    n1=50
    n2=50
    df_dict = read_files(args.input_dir)
    p=list(df_dict.values())[0].shape[1]-1
    
    lr = 1e-4
    momentum = 0.9
    hidden_features = p
    epochs = 10
    
    max_out_of_samples = 10
    bootstrap_size = 10
    
    model, total_loss_dict = pretraining(df_dict, args.y_column, args.mlp_layers, hidden_features = hidden_features, epochs = epochs, lr=lr, momentum=momentum)
    torch.save(model.state_dict(), args.output_model)
    
    feature_df_dict = convert_df_to_feature(df_dict, args.y_column, model)
    
    p=list(feature_df_dict.values())[0].shape[1]-1
    clusterer = GreedyClustering(feature_df_dict, args.y_column)
    cluster_dict, result = clusterer.clustering(n1, n2, p, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, threshold=args.threshold)
    
    error_calculator = ErrorCalculator(feature_df_dict, cluster_dict, args.y_column)
    separate_feature_error_sum = error_calculator.sum_of_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    clustered_feature_error_sum = error_calculator.sum_of_clustered_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    
    output_dict = {
        'n1': [n1], 
        'n2': [n2], 
        'p': [p], 
        'cluster': [cluster_dict],
        'threshold': [args.threshold],
        'separate_feature_oos_error_sum': [separate_feature_error_sum.cpu().detach().numpy()],
        'clustered_feature_oos_error_sum': [clustered_feature_error_sum.cpu().detach().numpy()],
        'lr': [lr],
        'momentum': [momentum],
        'epochs': [epochs],
        'hidden_features': [hidden_features],
    }
    output = pd.DataFrame.from_dict(output_dict)
    output.to_pickle(args.output_file)