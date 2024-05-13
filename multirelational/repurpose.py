import torch
import random
import numpy as np
from traineval import *

import argparse


def main():
    parser = argparse.ArgumentParser(description='Run THNs forecasting evaluation')
    parser.add_argument('--model', choices=['HAN', 'RGCN', 'HGT'], required=True, help='Choose model')
    parser.add_argument('--update', choices=['GRU', 'MLP', 'PE', 'NO-UPD'], required=True, help='Choose model')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e)
        parser.print_help()
        exit(1)
    
    snapshots = get_icews_dataset()
    
    model = args.model
    upd = args.update
    
    nrun=10
    results_avgpr = []
    results_mrr = []
    for _ in range(nrun):
        avgpr, mrr = training_to_temporal(snapshots, 256, 128, model, upd)
        results_avgpr.append(avgpr)
        results_mrr.append(mrr)
        
    # Open the file in write mode
    with open(f'results-rep/{model}_{upd}_avgpr.txt', 'w') as file:
        # Write each fruit to the file
        for result in results_avgpr:
            file.write(str(result) + '\n')
            
    # Open the file in write mode
    with open(f'results-rep/{model}_{upd}_mrr.txt', 'w') as file:
        # Write each fruit to the file
        for result in results_mrr:
            file.write(str(result) + '\n')
    

if __name__ == "__main__":
    main()