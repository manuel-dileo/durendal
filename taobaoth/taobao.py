import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from torch_geometric_temporal.nn.recurrent import GConvGRU,  EvolveGCNH

import random

import gc
import copy

from itertools import permutations

import pandas as pd

import torch_geometric.transforms as T

import networkx as nx
import numpy as np

import json

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src')

from durendal import *

from torch_geometric.nn import GCNConv, GATv2Conv, Linear, HANConv, HeteroConv


def get_taobao_dataset():
    user_x = torch.load('TAOBAO-5/user.pt')
    item_x = torch.load('TAOBAO-5/item.pt')
    category_x = torch.load('TAOBAO-5/category.pt')
    itc_edges = torch.load('TAOBAO-5/itc_edge_index.pt').long()
    
    bs = [('user','pageview','item'),\
          ('user','fav','item'),\
          ('user','cart','item'),\
          ('user','buy','item'),\
         ]
    
    snapshots = []
    for isnap in range(0,288):
        snap = HeteroData()
        snap['user'].x = user_x
        snap['item'].x = item_x
        snap['category'].x = category_x
        snap['item','to','category'].edge_index = itc_edges
        for b in bs:
            bvalue = torch.load(f'TAOBAO-5/{isnap}_{b}.pt')
            snap[b].edge_index = bvalue.long()
        snapshots.append(snap)
    return snapshots

class TAOBAODurendal(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(TAOBAODurendal, self).__init__()
        #user nodes have no in-edge so we need to project their features
        self.projuser1 = Linear(in_channels['user'], hidden_conv_1) 
        self.conv1 = DurendalConv(in_channels, hidden_conv_1, metadata)
        self.update1 = SemanticUpdateWA(n_channels=hidden_conv_1, tau=0.05)
        #self.update1 = SemanticUpdateGRU(n_channels=hidden_conv_1)
        #self.update1 = SemanticUpdateMLP(n_channels=hidden_conv_1)
        self.agg1 = SemanticAggregation(n_channels=hidden_conv_1)
        
        #user nodes have no in-edge so we need to project their features
        self.projuser2 = Linear(hidden_conv_1, hidden_conv_2)
        self.conv2 = DurendalConv(hidden_conv_1, hidden_conv_2, metadata)
        self.update2 = SemanticUpdateWA(n_channels=hidden_conv_2, tau=0.05)
        #self.update2 = SemanticUpdateGRU(n_channels=hidden_conv_2)
        #self.update2 = SemanticUpdateMLP(n_channels=hidden_conv_2)
        self.agg2 = SemanticAggregation(n_channels=hidden_conv_2)
        
        self.post = Linear(hidden_conv_2, 2)
            
        self.past_out_dict_1 = None
        self.past_out_dict_2 = None
        
        self.loss_fn = BCEWithLogitsLoss()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.update1.reset_parameters()
        self.agg1.reset_parameters()
        self.conv2.reset_parameters()
        self.update2.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, snap, past_out_dict_1=None, past_out_dict_2=None):
        
        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()
        
        
        out_dict = self.conv1(x_dict, edge_index_dict)
        for k, v in out_dict['user'].items():
            out_dict['user'][k] = self.projuser1(out_dict['user'][k])
        
        if snap==0:
            current_dict_1 = out_dict.copy()
        else: 
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()
        out_dict = self.agg1(current_dict_1)
        
        out_dict = self.conv2(out_dict, edge_index_dict)
        for k, v in out_dict['user'].items():
            out_dict['user'][k] = self.projuser2(out_dict['user'][k])
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = self.agg2(current_dict_2)
        
        #HADAMARD MLP
        h_src = out_dict['user'][edge_label_index[0]]
        h_dst = out_dict['item'][edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class TAOBAOATU(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(TAOBAOATU, self).__init__()
        self.conv1 = DurendalConv(in_channels, hidden_conv_1, metadata)
        self.projuser1 = Linear(in_channels['user'], hidden_conv_1)
        self.agg1 = SemanticAggregation(n_channels=hidden_conv_1)
        self.update1 = HetNodeUpdateGRU(hidden_conv_1, metadata)
        
        
        self.conv2 = DurendalConv(hidden_conv_1, hidden_conv_2, metadata)
        self.projuser2 = Linear(hidden_conv_1, hidden_conv_2)
        self.agg2 = SemanticAggregation(n_channels=hidden_conv_2)
        self.update2 = HetNodeUpdateGRU(hidden_conv_2, metadata)
        
        self.post = Linear(hidden_conv_2, 2)
            
        self.past_out_dict_1 = None
        self.past_out_dict_2 = None
        
        self.loss_fn = BCEWithLogitsLoss()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.update1.reset_parameters()
        self.agg1.reset_parameters()
        self.conv2.reset_parameters()
        self.update2.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, edge_label_index, snap, past_out_dict_1=None, past_out_dict_2=None):
        
        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()
            
        out_dict = self.conv1(x_dict, edge_index_dict)
        for k, v in out_dict['user'].items():
            out_dict['user'][k] = self.projuser1(out_dict['user'][k])
        out_dict = self.agg1(out_dict)
        if snap==0:
            current_dict_1 = out_dict.copy()
        else: 
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()
        
        out_dict = self.conv2(out_dict, edge_index_dict)
        for k, v in out_dict['user'].items():
            out_dict['user'][k] = self.projuser2(out_dict['user'][k])
        out_dict = self.agg2(out_dict)
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        
        #HADAMARD MLP
        h_src = out_dict['user'][edge_label_index[0]]
        h_dst = out_dict['item'][edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class TAOBAOGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(TAOBAOGAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_conv_1)
        self.conv2 = GATv2Conv(hidden_conv_1, hidden_conv_2)
                             
        self.post = Linear(hidden_conv_2, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x, edge_index, edge_label_index):
        out = self.conv1(x, edge_index)
        out = out.relu()
        out = self.conv2(out, edge_index)
        out = out.relu()
        
        #HADAMARD MLP
        h = out
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class TAOBAOHAN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_conv_1,
        hidden_conv_2,
        metadata
    ):
        super(TAOBAOHAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
        self.projuser1 = Linear(in_channels['user'], hidden_conv_1)
        self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)
        self.projuser2 = Linear(hidden_conv_1, hidden_conv_2)
                             
        self.post = Linear(hidden_conv_2, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict['user'] = x_dict['user'] #han drop no in edges node types
        out_dict['user'] = self.projuser1(out_dict['user'])
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        user_emb = copy.copy(out_dict['user'])
        out_dict = self.conv2(out_dict, edge_index_dict)
        out_dict['user'] = self.projuser2(user_emb)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        
        #HADAMARD MLP
        h_src = out_dict['user'][edge_label_index[0]]
        h_dst = out_dict['item'][edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

class TAOBAOGConvGRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_conv_2):
        super(TAOBAOGConvGRU, self).__init__()
        self.gcgru = GConvGRU(in_channels, hidden_conv_2, 2)
        self.post = torch.nn.Linear(hidden_conv_2, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, H=None):
        h = self.gcgru(x, edge_index, H=H)
        hidden = torch.Tensor(h.detach().numpy())
        h = F.relu(h)
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h, hidden
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class TAOBAOEvolveGCN(torch.nn.Module):
    def __init__(self, in_channels, num_nodes):
        super(TAOBAOEvolveGCN, self).__init__()
        self.evolve = EvolveGCNH(num_nodes, in_channels, add_self_loops=False)
        self.post = torch.nn.Linear(in_channels, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x, edge_index, edge_label_index):
        h = self.evolve(x, edge_index)
        h = F.relu(h)
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class TAOBAOHEGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes, 
        edge_types
    ):
        super(TAOBAOHEGCN, self).__init__()
          
        self.conv1 = HeteroConv({edge_t: EvolveGCNH(num_nodes, in_channels, add_self_loops=False) for edge_t in edge_types})
        self.post = torch.nn.Linear(in_channels, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        
        #HADAMARD MLP
        h_src = out_dict['user'][edge_label_index[0]]
        h_dst = out_dict['item'][edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)