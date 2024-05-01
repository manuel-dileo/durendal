import torch
import random
import numpy as np
from traineval import *

import argparse

load_dataset = {
    'steemitth': get_steemit_dataset,
    'gdelt18': get_gdelt_dataset,
    'icews18': get_icews_dataset
}

def main():
    parser = argparse.ArgumentParser(description='Run THNs forecasting evaluation')

    parser.add_argument('--seed', type=int, default=0, help='Seed value (default: 0)')
    parser.add_argument('--dataset', choices=['steemitth', 'gdelt18', 'icews18'], required=True, help='Choose dataset')
    parser.add_argument('--model', choices=['DURENDAL-UTA', 'DURENDAL-ATU', 'DyHAN', 'HTGNN', 'REGCN', 'HAN', 'HetEvolveGCN', 'complex', 'tntcomplex'], required=True, help='Choose model')

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e)
        parser.print_help()
        exit(1)
        
    seed = args.seed
    device = torch.device('cuda')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(12345)
    random.seed(12345)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    snapshots = load_dataset[args.dataset]()
    
    model = args.model
    if model == 'DURENDAL-UTA':
        training_durendal_uta(snapshots, hidden_conv_1=256, hidden_conv_2=128)
    elif model == 'DURENDAL-ATU':
        training_durendal_atu(snapshots, hidden_conv_1=256, hidden_conv_2=128)
    elif model == 'DyHAN':
        training_dyhan(snapshots, hidden_conv_1=256, hidden_conv_2=128)
    elif model == 'HTGNN':
        training_htgnn(snapshots, hidden_conv_1=32, hidden_conv_2=16)
    elif model == 'REGCN':
        training_regcn(snapshots, hidden_conv_1=32, hidden_conv_2=16, output_conv=8)
    elif model == 'HAN':
        training_han(snapshots, hidden_conv_1=256, hidden_conv_2=128)
    elif model == 'HetEvolveGCN':
        training_hev(snapshots, hidden_conv_1=256, hidden_conv_2=128)
    elif model == 'complex':
        training_complex(snapshots)
    else:
        training_tntcomplex(snapshots)  

    

if __name__ == "__main__":
    main()