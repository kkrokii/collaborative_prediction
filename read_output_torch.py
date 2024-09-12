import pandas as pd
import argparse
import torch

def read_output_file(output_dir, file_name):
    output = pd.read_pickle(output_dir + '/' + file_name).to_dict('records')[0]
    
    print(f"n1: {output['n1']}, n2: {output['n2']}, p: {output['p']}, correct: {output['correct']}, total test samples: {len(output['gt_list_for_distinct_distr']) + len(output['gt_list_for_identical_distr'])}")
    print(f"\t distinct distribution pairs for training: {output['distinct_distributions_train']}, identical distribution pairs for training: {output['identical_distributions_train']}")
    print(f"\t bound type: {output['bound_type']}, alpha: {output['alpha']}, beta perturbation: {output['beta_perturbation_level'] if output['beta_perturbation_level'] is not None else output['beta_setting']}")
    print("combine datasets if (LHS bound) > (RHS bound)")
    
    print("distinct distribution (datasets should not be combined): ")
    for i in range(5):
        print(f"\t LHS bound: {round(output['lhs_bound_list_for_distinct_distr'][i].item(),4)},", end='')
        print(f"RHS bound: {round(output['rhs_bound_list_for_distinct_distr'][i].item(),4)},", end='')
        print(f"\t LHS bound > RHS bound: {(output['lhs_bound_list_for_distinct_distr'][i]>output['rhs_bound_list_for_distinct_distr'][i]).item()}, GT: {output['gt_list_for_distinct_distr'][i].item()}")
    print(f"\t overall GT: {True if sum(output['gt_list_for_distinct_distr']) > len(output['gt_list_for_distinct_distr'])-sum(output['gt_list_for_distinct_distr']) else False} ", end='')
    print(f"(True: {sum(output['gt_list_for_distinct_distr']).item()}, False: {(len(output['gt_list_for_distinct_distr'])-sum(output['gt_list_for_distinct_distr'])).item()})")
        
    print("identical distribution (datasets should be combined): ")
    for i in range(5):
        print(f"\t LHS bound: {round(output['lhs_bound_list_for_identical_distr'][i].item(),4)}, ", end='')
        print(f"RHS bound: {round(output['rhs_bound_list_for_identical_distr'][i].item(),4)}, ", end='')
        print(f"\t LHS bound > RHS bound: {(output['lhs_bound_list_for_identical_distr'][i]>output['rhs_bound_list_for_identical_distr'][i]).item()}, GT: {output['gt_list_for_identical_distr'][i].item()}")
    print(f"\t overall GT: {True if sum(output['gt_list_for_identical_distr']) > len(output['gt_list_for_identical_distr'])-sum(output['gt_list_for_identical_distr']) else False} ", end='')
    print(f"(True: {sum(output['gt_list_for_identical_distr']).item()}, False: {(len(output['gt_list_for_identical_distr'])-sum(output['gt_list_for_identical_distr'])).item()})")

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
python read_output_torch.py --output_dir=output_with_errors/50,50,10,500,500/bound2 --output_file=output2.pd
'''