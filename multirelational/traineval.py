import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

import random

import bisect

import gc
import copy

from itertools import permutations

import pandas as pd

from torch_geometric.utils import negative_sampling, structured_negative_sampling
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit,NormalizeFeatures,Constant,OneHotDegree
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear, GCN, GAT

import torch
import networkx as nx
import numpy as np

import copy

from multirelational import *

def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    
    Function useful to compute MRR.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)
    return lo

def compute_mrr(real_scores, fake_scores):
    srr = 0
    count = 0
    for i,score in enumerate(real_scores):
        try:
            fake_scores_cp = copy.copy([fake_scores[i]])
        except IndexError: break
        rank = reverse_insort(fake_scores_cp, score)
        rr = 1/(rank+1) #index starts from zero
        srr+=rr
        count+=1
    return srr/count


def durendal_train_single_snapshot(model, data, i_snap, train_data, val_data, test_data,\
                          past_dict_1, past_dict_2,\
                          optimizer, device='cpu', num_epochs=50, verbose=False):
    
    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    best_past_dict_1 = {}
    best_past_dict_2 = {}
    
    tol = 5e-2
    
    edge_types = list(data.edge_index_dict.keys())
    
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
        
        pred_dict, past_dict_1, past_dict_2 =\
            model(train_data.x_dict, train_data.edge_index_dict, train_data,\
                  i_snap, past_dict_1, past_dict_2)
        
        preds = torch.Tensor()
        edge_labels = torch.Tensor()
        for edge_t in edge_types:
            preds = torch.cat((preds,pred_dict[edge_t]),-1)
            edge_labels = torch.cat((edge_labels,train_data[edge_t].edge_label.type_as(pred_dict[edge_t])),-1)
            
        #compute loss function based on all edge types
        loss = model.loss(preds, edge_labels)
        loss = torch.autograd.Variable(loss, requires_grad = True)
        
        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val, mrr_val = durendal_test(model, i_snap, val_data, data, device)
        
        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        gc.collect()
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_past_dict_1 = past_dict_1
            best_past_dict_2 = past_dict_2
            best_model = model
        else:
            break
        
    avgpr_score_test, mrr_test = durendal_test(model, i_snap, test_data, data, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, avgpr_score_test, mrr_test, best_past_dict_1, best_past_dict_2, optimizer


def regcn_train_single_snapshot(model, data, i_snap, train_data, val_data, test_data, \
                                   past_dict_1, past_dict_2, past_r,\
                                   optimizer, device='cpu', num_epochs=50, verbose=False):

    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    best_past_dict_1 = {}
    best_past_dict_2 = {}
    best_past_r = None

    tol = 5e-2

    edge_types = list(data.edge_index_dict.keys())


    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()

        pred_dict, past_dict_1, past_dict_2, past_r = \
            model(train_data.x_dict, train_data.edge_index_dict, train_data, \
                  i_snap, past_dict_1, past_dict_2, past_r)

        preds = torch.Tensor()
        edge_labels = torch.Tensor()
        for edge_t in edge_types:
            preds = torch.cat((preds,pred_dict[edge_t]),-1)
            edge_labels = torch.cat((edge_labels,train_data[edge_t].edge_label.type_as(pred_dict[edge_t])),-1)

        #compute loss function based on all edge types
        loss = model.loss(preds, edge_labels)
        loss = torch.autograd.Variable(loss, requires_grad = True)

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val, mrr_val = durendal_test(model, i_snap, val_data, data, device)

        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        gc.collect()
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_past_dict_1 = past_dict_1
            best_past_dict_2 = past_dict_2
            best_past_r = past_r
            best_model = model
        else:
            break

    avgpr_score_test, mrr_test = durendal_test(model, i_snap, test_data, data, device)

    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')

    return best_model, avgpr_score_test, mrr_test, best_past_dict_1, best_past_dict_2, best_past_r, optimizer


def het_train_single_snapshot(model, data, train_data, val_data, test_data,\
                          optimizer, device='cpu', num_epochs=50, verbose=False):
    
    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    
    tol = 5e-2
    
    edge_types = list(data.edge_index_dict.keys())
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
            
        pred_dict =\
            model(train_data.x_dict, train_data.edge_index_dict, train_data)
        
        preds = torch.Tensor()
        edge_labels = torch.Tensor()
        for edge_t in edge_types:
            preds = torch.cat((preds,pred_dict[edge_t]),-1)
            edge_labels = torch.cat((edge_labels,train_data[edge_t].edge_label.type_as(pred_dict[edge_t])),-1)
        
        #compute loss function based on all edge types
        loss = model.loss(preds, edge_labels)
        loss = torch.autograd.Variable(loss, requires_grad = True)
        
        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val, mrr_val = het_test(model, val_data, data, device)
        
        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        gc.collect()
        
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_model = model
        else:
            break
        
        
    avgpr_score_test, mrr_test = het_test(model, test_data, data, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, avgpr_score_test, mrr_test, optimizer

def tnt_train_single_snapshot(model, data, isnap, train_data, val_data, test_data,\
                              optimizer, device='cpu', num_epochs=50, verbose=False):
    
    
    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    
    tol = 5e-2
    
    edge_types = list(data.edge_index_dict.keys())
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
            
        pred_dict =\
            model(train_data.x_dict, train_data.edge_index_dict, train_data, isnap)
        
        preds = torch.Tensor()
        edge_labels = torch.Tensor()
        for edge_t in edge_types:
            preds = torch.cat((preds,pred_dict[edge_t]),-1)
            edge_labels = torch.cat((edge_labels,train_data[edge_t].edge_label.type_as(pred_dict[edge_t])),-1)
        
        #compute loss function based on all edge types
        loss = model.loss(preds, edge_labels)
        loss = torch.autograd.Variable(loss, requires_grad = True)
        
        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val, mrr_val = tnt_test(model, isnap, val_data, data, device)
        
        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_model = model
        else:
            break
        
    avgpr_score_test, mrr_test = tnt_test(model, isnap, test_data, data, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, avgpr_score_test, mrr_test, optimizer

def durendal_test(model, i_snap, test_data, data, device='cpu'):
    
    model.eval()

    test_data = test_data.to(device)
    edge_types = list(data.edge_index_dict.keys())
    

    h_dict, *_ = model(test_data.x_dict, test_data.edge_index_dict, test_data, i_snap)
    
    tot_avgpr = 0
    tot_mrr = 0
    
    num_rel = 0
    
    for edge_t in edge_types:
        
        h = h_dict[edge_t]
        pred_cont = torch.sigmoid(h).cpu().detach().numpy()
        
        num_pos = (len(test_data[edge_t].edge_label_index[0])//2)
        h_fake = h[num_pos:]
        
        fake_preds = torch.sigmoid(h_fake).cpu().detach().numpy()
        edge_label = test_data[edge_t].edge_label.cpu().detach().numpy()
      
        if len(edge_label) >0:
            avgpr_score = average_precision_score(edge_label, pred_cont)
            mrr_score = compute_mrr(pred_cont[:num_pos], fake_preds)
        
            tot_avgpr += avgpr_score
            tot_mrr += mrr_score
            num_rel +=1

    return tot_avgpr/num_rel, tot_mrr/num_rel


def tnt_test(model, isnap, test_data, data, device='cpu'):
        
    model.eval()

    test_data = test_data.to(device)
    
    edge_types = list(data.edge_index_dict.keys())

    h_dict = model(test_data.x_dict, test_data.edge_index_dict, test_data, isnap)
    
    tot_avgpr = 0
    tot_mrr = 0
    
    num_rel = 0
    
    for edge_t in edge_types:
        
        h = h_dict[edge_t]
        pred_cont = torch.sigmoid(h).cpu().detach().numpy()
        
        num_pos = (len(test_data[edge_t].edge_label_index[0])//2)
        h_fake = h[num_pos:]
        
        fake_preds = torch.sigmoid(h_fake).cpu().detach().numpy()
        edge_label = test_data[edge_t].edge_label.cpu().detach().numpy()
      
        if len(edge_label) >0:
            avgpr_score = average_precision_score(edge_label, pred_cont)
            mrr_score = compute_mrr(pred_cont[:num_pos], fake_preds)
        
            tot_avgpr += avgpr_score
            tot_mrr += mrr_score
            num_rel +=1

    return tot_avgpr/num_rel, tot_mrr/num_rel

def het_test(model, test_data, data, device='cpu'):
        
    model.eval()

    test_data = test_data.to(device)
    
    edge_types = list(data.edge_index_dict.keys())

    h_dict = model(test_data.x_dict, test_data.edge_index_dict, test_data)
    
    tot_avgpr = 0
    tot_mrr = 0
    
    num_rel = 0
    
    for edge_t in edge_types:
        
        h = h_dict[edge_t]
        pred_cont = torch.sigmoid(h).cpu().detach().numpy()
        
        num_pos = (len(test_data[edge_t].edge_label_index[0])//2)
        h_fake = h[num_pos:]
        
        fake_preds = torch.sigmoid(h_fake).cpu().detach().numpy()
        edge_label = test_data[edge_t].edge_label.cpu().detach().numpy()
      
        if len(edge_label) >0:
            avgpr_score = average_precision_score(edge_label, pred_cont)
            mrr_score = compute_mrr(pred_cont[:num_pos], fake_preds)
        
            tot_avgpr += avgpr_score
            tot_mrr += mrr_score
            num_rel +=1

    return tot_avgpr/num_rel, tot_mrr/num_rel

def training_durendal_uta(snapshots, hidden_conv_1, hidden_conv_2, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())
    
    lr = 0.001
    weight_decay = 5e-3
    
    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    #DURENDAL
    durendal = RDurendal(in_channels, num_nodes, hetdata.metadata(),
                        hidden_conv_1=hidden_conv_1,
                        hidden_conv_2=hidden_conv_2)
    
    durendal.reset_parameters()
    
    durendalopt = torch.optim.Adam(params=durendal.parameters(), lr=lr, weight_decay = weight_decay)
    
    
    past_dict_1 = {}
    for node in hetdata.x_dict.keys():
        past_dict_1[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_1[src][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[src].num_nodes)])
        past_dict_1[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[dst].num_nodes)])
        
    past_dict_2 = {}
    for node in hetdata.x_dict.keys():
        past_dict_2[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_2[src][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[src].num_nodes)])
        past_dict_2[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[dst].num_nodes)])
    
    
    durendal_avgpr = 0
    durendal_mrr = 0
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        
        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)
        
        het_train_data, _, het_val_data = link_split(snapshot)
     
        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        durendal, dur_avgpr_test, dur_mrr_test , past_dict_1, past_dict_2, durendalopt =\
            durendal_train_single_snapshot(durendal, snapshot, i, het_train_data, het_val_data, het_test_data,\
                                  past_dict_1, past_dict_2, durendalopt)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' DURENDAL AVGPR Test: {dur_avgpr_test} \n MRR Test: {dur_mrr_test}\n')
        durendal_avgpr += dur_avgpr_test
        durendal_mrr += dur_mrr_test
        
        gc.collect()
        
        
    durendal_avgpr_all = durendal_avgpr / (num_snap-1)
    durendal_mrr_all = durendal_mrr / (num_snap-1)
    
    print('DURENDAL')
    print(f'\tAVGPR over time: Test: {durendal_avgpr_all}')
    print(f'\tMRR over time: Test: {durendal_mrr_all}')
    print()
    
    return

def training_dyhan(snapshots, hidden_conv_1, hidden_conv_2, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())
    
    lr = 0.001
    weight_decay = 5e-3
    
    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    #DyHAN
    dyhan = DyHAN(in_channels, num_nodes, hetdata.metadata(),
                        hidden_conv_1=hidden_conv_1,
                        hidden_conv_2=hidden_conv_2)
    
    dyhan.reset_parameters()
    
    dyhanopt = torch.optim.Adam(params=dyhan.parameters(), lr=lr, weight_decay = weight_decay)
    
    past_dict_1_dyhan = {}
    for node in hetdata.x_dict.keys():
        past_dict_1_dyhan[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_1_dyhan[src][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[src].num_nodes)])
        past_dict_1_dyhan[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[dst].num_nodes)])
        
    past_dict_2_dyhan = {}
    for node in hetdata.x_dict.keys():
        past_dict_2_dyhan[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_2_dyhan[src][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[src].num_nodes)])
        past_dict_2_dyhan[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[dst].num_nodes)])
    
    dyhan_mrr = 0
    dyhan_avgpr = 0
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        
        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)
        
        het_train_data, _, het_val_data = link_split(snapshot)
     
        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        dyhan, dyhan_avgpr_test, dyhan_mrr_test , past_dict_1_dyhan, past_dict_2_dyhan, dyhanopt =\
            durendal_train_single_snapshot(dyhan, snapshot, i, het_train_data, het_val_data, het_test_data,\
                                  past_dict_1_dyhan, past_dict_2_dyhan, dyhanopt, num_epochs=3)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' DyHAN AVGPR Test: {dyhan_avgpr_test} \n MRR Test: {dyhan_mrr_test}\n')
        dyhan_avgpr += dyhan_avgpr_test
        dyhan_mrr += dyhan_mrr_test
        gc.collect()
        
    dyhan_avgpr_all = dyhan_avgpr / (num_snap-1)
    dyhan_mrr_all = dyhan_mrr / (num_snap-1)
    
    print('DyHAN')
    print(f'\tAVGPR over time: Test: {dyhan_avgpr_all}')
    print(f'\tMRR over time: Test: {dyhan_mrr_all}')
    print()
    
    return

def training_han(snapshots, hidden_conv_1, hidden_conv_2, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())
    
    lr = 0.001
    weight_decay = 5e-3
    
    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    #HAN
    han = RHAN(in_channels, hidden_conv_1, hidden_conv_2, hetdata.metadata())
    han.reset_parameters()
    hanopt = torch.optim.Adam(params=han.parameters(), lr=lr, weight_decay = weight_decay)
    
    han_avgpr = 0
    han_mrr = 0
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        
        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)
        
        het_train_data, _, het_val_data = link_split(snapshot)
     
        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        han, han_avgpr_test, han_mrr_test, hanopt =\
            het_train_single_snapshot(han, snapshot, het_train_data, het_val_data, het_test_data, hanopt)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' HAN AVGPR Test: {han_avgpr_test} \n MRR Test: {han_mrr_test}\n')
        han_avgpr += han_avgpr_test
        han_mrr += han_mrr_test
        
        gc.collect()
        
    han_avgpr_all = han_avgpr / (num_snap-1)
    han_mrr_all = han_mrr / (num_snap-1)
    
    print('HAN')
    print(f'\tAVGPR over time: Test: {han_avgpr_all}')
    print(f'\tMRR over time: Test: {han_mrr_all}')
    print()
    
    return

def training_hev(snapshots, hidden_conv_1, hidden_conv_2, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())
    
    lr = 0.001
    weight_decay = 5e-3
    
    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    homdata = copy.deepcopy(snapshots[0]).to_homogeneous()
    in_channels_homo = homdata.x.size(1)
    num_nodes_homo = homdata.x.size(0)
    
    #HetEvolveGCN
    hev = RHEGCN(in_channels_homo, num_nodes_homo, list(hetdata.edge_index_dict.keys()))
    hev.reset_parameters()
    hevopt = torch.optim.Adam(params=hev.parameters(), lr=lr, weight_decay = weight_decay)
    
    hev_avgpr = 0
    hev_mrr = 0
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        
        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)
        
        het_train_data, _, het_val_data = link_split(snapshot)
     
        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP       
        hev, hev_avgpr_test, hev_mrr_test, hevopt =\
            het_train_single_snapshot(hev, snapshot, het_train_data, het_val_data, het_test_data, hevopt)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' HetEvolveGCN AVGPR Test: {hev_avgpr_test} \n MRR Test: {hev_mrr_test}\n')
        hev_avgpr += hev_avgpr_test
        hev_mrr += hev_mrr_test
        
        gc.collect()
        
    hev_avgpr_all = hev_avgpr / (num_snap-1)
    hev_mrr_all = hev_mrr / (num_snap-1)
    
    print('HetEvolveGCN')
    print(f'\tAVGPR over time: Test: {hev_avgpr_all}')
    print(f'\tMRR over time: Test: {hev_mrr_all}')
    
    return

def training_durendal_atu(snapshots, hidden_conv_1, hidden_conv_2, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())
    
    lr = 0.001
    weight_decay = 5e-3
    
    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    #ATU
    atu = RATU(in_channels, num_nodes, hetdata.metadata(),
                        hidden_conv_1=hidden_conv_1,
                        hidden_conv_2=hidden_conv_2)
    atu.reset_parameters()
    atuopt = torch.optim.Adagrad(params=atu.parameters(), lr=1e-1, weight_decay = weight_decay)
    
    past_dict_1_atu = {}
    for node in hetdata.x_dict.keys():
        past_dict_1_atu[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_1_atu[src][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[src].num_nodes)])
        past_dict_1_atu[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[dst].num_nodes)])
        
    past_dict_2_atu = {}
    for node in hetdata.x_dict.keys():
        past_dict_2_atu[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_2_atu[src][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[src].num_nodes)])
        past_dict_2_atu[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[dst].num_nodes)])
    
    atu_avgpr = 0
    atu_mrr = 0
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        
        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)
        
        het_train_data, _, het_val_data = link_split(snapshot)
     
        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)
        
        atu, atu_avgpr_test, atu_mrr_test , past_dict_1_atu, past_dict_2_atu, atuopt =\
            durendal_train_single_snapshot(atu, snapshot, i, het_train_data, het_val_data, het_test_data,\
                                  past_dict_1_atu, past_dict_2_atu, atuopt)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' ATU AVGPR Test: {atu_avgpr_test} \n MRR Test: {atu_mrr_test}\n')
        atu_avgpr += atu_avgpr_test
        atu_mrr += atu_mrr_test
        
        gc.collect()
        
    atu_avgpr_all = atu_avgpr / (num_snap-1)
    atu_mrr_all = atu_mrr / (num_snap-1)
    
    print('ATU')
    print(f'\tAVGPR over time: Test: {atu_avgpr_all}')
    print(f'\tMRR over time: Test: {atu_mrr_all}')
    
    return

def training_htgnn(snapshots, hidden_conv_1, hidden_conv_2, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())

    lr = 0.001
    weight_decay = 5e-3

    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    homdata = copy.deepcopy(snapshots[0]).to_homogeneous()
    in_channels_homo = homdata.x.size(1)
    num_nodes_homo = homdata.x.size(0)
    homdata=None

    #HTGNN
    htgnn = HTGNN(in_channels, in_channels_homo, num_nodes, hetdata.metadata(),
               hidden_conv_1=hidden_conv_1,
               hidden_conv_2=hidden_conv_2)
    htgnn.reset_parameters()
    htgnnopt = torch.optim.Adagrad(params=htgnn.parameters(), lr=1e-1, weight_decay = weight_decay)

    past_dict_1_htgnn = {}
    for node in hetdata.x_dict.keys():
        past_dict_1_htgnn[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_1_htgnn[src][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[src].num_nodes)])
        past_dict_1_htgnn[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[dst].num_nodes)])

    past_dict_2_htgnn = {}
    for node in hetdata.x_dict.keys():
        past_dict_2_htgnn[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_2_htgnn[src][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[src].num_nodes)])
        past_dict_2_htgnn[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[dst].num_nodes)])

    htgnn_avgpr = 0
    htgnn_mrr = 0

    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])

        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)

        het_train_data, _, het_val_data = link_split(snapshot)

        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)

        htgnn, htgnn_avgpr_test, htgnn_mrr_test , past_dict_1_htgnn, past_dict_2_htgnn, htgnnopt = \
            durendal_train_single_snapshot(htgnn, snapshot, i, het_train_data, het_val_data, het_test_data, \
                                           past_dict_1_htgnn, past_dict_2_htgnn, htgnnopt)

        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f'HTGNN AVGPR Test: {htgnn_avgpr_test} \n MRR Test: {htgnn_mrr_test}\n')
        htgnn_avgpr += htgnn_avgpr_test
        htgnn_mrr += htgnn_mrr_test

        gc.collect()

    htgnn_avgpr_all = htgnn_avgpr / (num_snap-1)
    htgnn_mrr_all = htgnn_mrr / (num_snap-1)

    print('HTGNN')
    print(f'\tAVGPR over time: Test: {htgnn_avgpr_all}')
    print(f'\tMRR over time: Test: {htgnn_mrr_all}')

    return

def training_regcn(snapshots, hidden_conv_1, hidden_conv_2, output_conv, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())

    lr = 0.001
    weight_decay = 5e-3

    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    homdata = copy.deepcopy(snapshots[0]).to_homogeneous()
    in_channels_homo = homdata.x.size(1)
    num_nodes_homo = homdata.x.size(0)
    homdata = None

    #ATU
    regcn = REGCN(in_channels_homo, num_nodes, hetdata.metadata(),
               hidden_conv_1=hidden_conv_1,
               hidden_conv_2=hidden_conv_2,
               output_conv = output_conv)

    regcn.reset_parameters()
    regcnopt = torch.optim.Adagrad(params=regcn.parameters(), lr=1e-1, weight_decay = weight_decay)

    past_dict_1_regcn = {}
    for node in hetdata.x_dict.keys():
        past_dict_1_regcn[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_1_regcn[src][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[src].num_nodes)])
        past_dict_1_regcn[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_1)] for i in range(hetdata[dst].num_nodes)])

    past_dict_2_regcn = {}
    for node in hetdata.x_dict.keys():
        past_dict_2_regcn[node] = {}
    for src,r,dst in hetdata.edge_index_dict.keys():
        past_dict_2_regcn[src][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[src].num_nodes)])
        past_dict_2_regcn[dst][r] = torch.Tensor([[0 for j in range(hidden_conv_2)] for i in range(hetdata[dst].num_nodes)])

    past_r = torch.randn(num_nodes_homo, hidden_conv_2)
    regcn_avgpr = 0
    regcn_mrr = 0

    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])

        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)

        het_train_data, _, het_val_data = link_split(snapshot)

        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)

        regcn, regcn_avgpr_test, regcn_mrr_test , past_dict_1_regcn, past_dict_2_regcn, past_r, regcnopt = \
            regcn_train_single_snapshot(regcn, snapshot, i, het_train_data, het_val_data, het_test_data,\
                                           past_dict_1_regcn, past_dict_2_regcn, past_r, regcnopt)

        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' RE-GCN AVGPR Test: {regcn_avgpr_test} \n MRR Test: {regcn_mrr_test}\n')
        regcn_avgpr += regcn_avgpr_test
        regcn_mrr += regcn_mrr_test

        gc.collect()

    regcn_avgpr_all = regcn_avgpr / (num_snap-1)
    regcn_mrr_all = regcn_mrr / (num_snap-1)

    print('RE-GCN')
    print(f'\tAVGPR over time: Test: {regcn_avgpr_all}')
    print(f'\tMRR over time: Test: {regcn_mrr_all}')

    return

def training_complex(snapshots, dimension=2000, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())
    
    lr = 0.001
    weight_decay = 5e-3
    
    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    homdata = copy.deepcopy(snapshots[0]).to_homogeneous()
    in_channels_homo = homdata.x.size(1)
    num_nodes_homo = homdata.x.size(0)
    
    #ComplEx
    cplex = ComplEx(dimension, num_nodes_homo, hetdata.metadata())
    cplexopt = torch.optim.Adam(params=cplex.parameters(), lr=lr, weight_decay = weight_decay)
    
    cplex_avgpr = 0
    cplex_mrr = 0
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        
        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)
        
        het_train_data, _, het_val_data = link_split(snapshot)
     
        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        cplex, cplex_avgpr_test, cplex_mrr_test, cplexopt =\
            het_train_single_snapshot(cplex, snapshot, het_train_data, het_val_data, het_test_data, cplexopt)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' ComplEx AVGPR Test: {cplex_avgpr_test} \n MRR Test: {cplex_mrr_test}\n')
        cplex_avgpr += cplex_avgpr_test
        cplex_mrr += cplex_mrr_test
        
        
    cplex_avgpr_all = cplex_avgpr / (num_snap-1)
    cplex_mrr_all = cplex_mrr / (num_snap-1)
    
    print('ComplEx')
    print(f'\tAVGPR over time: Test: {cplex_avgpr_all}')
    print(f'\tMRR over time: Test: {cplex_mrr_all}')
    
    return


def training_tntcomplex(snapshots, dimension, rnn_size=500, device='cpu'):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())
    
    lr = 0.001
    weight_decay = 5e-3
    
    in_channels = {node: len(v[0]) for node,v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}
    
    homdata = copy.deepcopy(snapshots[0]).to_homogeneous()
    in_channels_homo = homdata.x.size(1)
    num_nodes_homo = homdata.x.size(0)
    
    #TNTComplEx
    tnt = TNTComplEx(2000, num_nodes_homo, hetdata.metadata(), len(snapshots), rnn_size)
    tntopt = torch.optim.Adam(params=tnt.parameters(), lr=lr, weight_decay = weight_decay)
    
    tnt_avgpr = 0
    tnt_mrr = 0
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        
        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)
        
        het_train_data, _, het_val_data = link_split(snapshot)
     
        het_test_data = copy.deepcopy(snapshots[i+1])
        future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        het_test_data, _, _ = future_link_split(het_test_data)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        tnt, tnt_avgpr_test, tnt_mrr_test, tntopt =\
            tnt_train_single_snapshot(tnt, snapshot, i, het_train_data, het_val_data, het_test_data, tntopt)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n')
        print(f' TNTComplEx AVGPR Test: {tnt_avgpr_test} \n MRR Test: {tnt_mrr_test}\n')
        tnt_avgpr += tnt_avgpr_test
        tnt_mrr += tnt_mrr_test
        
    tnt_avgpr_all = tnt_avgpr / (num_snap-1)
    tnt_mrr_all = tnt_mrr / (num_snap-1)
    
    print('TNTComplEx')
    print(f'\tAVGPR over time: Test: {tnt_avgpr_all}')
    print(f'\tMRR over time: Test: {tnt_mrr_all}')
    
    return