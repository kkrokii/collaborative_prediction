from real_data import pairwise_comparison, DataGenerator, split_X_y_and_tensorize
import numpy as np
import pandas as pd
import argparse
import os
import copy

import torch

def read_files(input_dir):
    file_dict = {f"{f}":pd.read_csv(os.path.join(input_dir, f)) for f in os.listdir(input_dir) if os.path.isfile( os.path.join(input_dir, f) )}
    return file_dict

def merge_datasets(df1, df2):
    return pd.concat([df1, df2])

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
        cluster = {k:-1 for k in self.df_dict.keys()}
        result = {k:None for k in self.df_dict.keys()}
        candidate_df_dict = copy.deepcopy(self.df_dict)
        total_clusters = 0
        for df_key, df in self.df_dict.items():
            if cluster[df_key] != -1:
                continue
            total_clusters += 1
            cluster[df_key] = total_clusters
            del candidate_df_dict[df_key]
            merged_df_key_list, merged_result_dict = self.greedy(n1, n2, p, df_key, df, candidate_df_dict, self.y_column, train_ratio, estimation_ratio, test_ratio, threshold)
            
            for k in merged_df_key_list:
                cluster[k] = total_clusters
                result[k] = merged_result_dict
                del candidate_df_dict[k]
            result[df_key] = merged_result_dict
        return cluster, result



class ErrorCalculator(DataGenerator):
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
        
    def out_of_sample_error(self, df, beta_hat, max_out_of_samples=1000):
        X0, y0 = self.generate_samples(df = df, samples=1, n=max_out_of_samples) 
        return (torch.mean((y0 - X0 @ beta_hat)**2, dim=-2)).squeeze()           
    
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
            sum_of_oos += self.out_of_sample_error(df, clustered_beta_hat[ self.cluster[df_name] ], max_out_of_samples)
        return sum_of_oos
    
    def sum_of_clustered_out_of_sample_error(self, n, max_out_of_samples=1000, bootstrap_size = 1000):
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
        '--threshold',
        type=float,
        default=0.9
    )
    args = parser.parse_args()
    
    n1=50
    n2=50
    df_dict = read_files(args.input_dir)
    p=list(df_dict.values())[0].shape[1]-1
    max_out_of_samples = 1000
    bootstrap_size = 1000
    
    clusterer = GreedyClustering(df_dict, args.y_column)
    cluster, result = clusterer.clustering(n1, n2, p, train_ratio=0.4, estimation_ratio=0.4, test_ratio=0.2, threshold=args.threshold)
    error_calculator = ErrorCalculator(df_dict, cluster, args.y_column)
    separate_error_sum = error_calculator.sum_of_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    clustered_error_sum = error_calculator.sum_of_clustered_out_of_sample_error(n1, max_out_of_samples=max_out_of_samples, bootstrap_size=bootstrap_size)
    
    output = pd.DataFrame.from_dict({
        'n1': [n1], 
        'n2': [n2], 
        'p': [p], 
        'cluster': [cluster],
        'threshold': [args.threshold],
        'separate_oos_error_sum': [separate_error_sum.cpu()],
        'clustered_oos_error_sum': [clustered_error_sum.cpu()],
    })
    output.to_pickle(args.output_file)