import pandas as pd
import argparse
import torch

def read_output_file(output_dir, file_name):
    output = pd.read_pickle(output_dir + '/' + file_name).to_dict('records')[0]
    
    print(f"n1: {output['n1']}, n2: {output['n2']}, p: {output['p']}, correct: {output['correct']}, total test samples: {len(output['gt_list'])}")
    print(f"\t bound type: {output['bound_type']}, alpha: {output['alpha']}")
    print("combine datasets if (LHS bound) > (RHS bound)")
    
    for i in range(5):
        print(f"\t LHS bound: {round(output['lhs_bound_list'][i].item(),4)},", end='')
        print(f"RHS bound: {round(output['rhs_bound_list'][i].item(),4)},", end='')
        print(f"\t LHS bound > RHS bound: {(output['lhs_bound_list'][i]>output['rhs_bound_list'][i]).item()}, GT: {output['gt_list'][i].item()}")
    gt_trues = sum(output['gt_list'])
    gt_falses = len(output['gt_list']) - gt_trues
    print(f"\t overall GT: {True if gt_trues > gt_falses else False} ", end='')
    print(f"(True: {gt_trues.item()}, False: {gt_falses.item()})")    
    
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
python read_real_output_torch.py --output_dir=data2/output --output_file=output2.pd
'''