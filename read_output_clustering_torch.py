import pandas as pd
import argparse
import torch

def read_output_file(output_dir, file_name):
    output = pd.read_pickle(output_dir + '/' + file_name).to_dict('records')[0]
    
    print(f"n1: {output['n1']}, n2: {output['n2']}, p: {output['p']}, bound type: {output['bound_type']}, threshold: {output['threshold']}")
    print(f"cluster: {output['cluster']}")
    print(f"sum of out of errors: \n\t separate models: {output['separate_oos_error_sum']}, clustered models: {output['clustered_oos_error_sum']}")
    
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
    args = parser.parse_args()
    read_output_file(args.output_dir, args.output_file)
'''
ex)
python read_output_clustering_torch.py --output_dir=cluster_output/data2 --output_file=output1.pd
'''