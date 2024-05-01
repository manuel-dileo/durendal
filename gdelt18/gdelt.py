import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import GDELT

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


def get_gdelt_dataset():
    dataset = GDELT(root='GDELT/')
    #num_rel = len(set([int(data.rel) for data in dataset]))
    num_rel = 238
    #last_snap = len(t_dict.keys())
    last_snap = 2138
    t_dict = {}
    for link in dataset:
        t = int(link.t)
        if t not in t_dict.keys():
            t_dict[t] = []
        t_dict[t].append(link)
    snapshots = []
    max_node = -1
    isnap=1
    top20 = ['16', '2', '3', '0', '19', '7', '34', '23', '6', '18', '30', '17', '15', '24', '11', '9', '1', '37', '4', '20'] #top 20 edge types for num of edges, previously computed
    for t, edges in t_dict.items():
        snap = HeteroData()
        edge_index_dict = {('node',r,'node'): [[],[]] for r in top20}
        for edge in edges:
            sub = int(edge.sub)
            rel = str(int(edge.rel))
            obj = int(edge.obj)
            if rel not in top20: continue
            num_node = max(sub, obj)
            if isnap<672 and num_node > max_node: #672: 1 week (24*4*7)
                max_node = num_node
            if isnap<672:
                edge_index_dict['node',f'{rel}','node'][0].append(sub)
                edge_index_dict['node',f'{rel}','node'][1].append(obj)
            elif sub > max_node or obj > max_node: pass
            else:
                edge_index_dict['node',f'{rel}','node'][0].append(sub)
                edge_index_dict['node',f'{rel}','node'][1].append(obj)
            
        if isnap==last_snap:
            for edge_t, edge_index in edge_index_dict.items():
                snap[edge_t].edge_index = torch.Tensor(edge_index).long()
            snap['node'].x = torch.Tensor([[1] for i in range(max_node)])
            snapshots.append(snap)
            break
    
        if isnap%672!=0: 
            isnap+=1
            continue
        for edge_t, edge_index in edge_index_dict.items():
            snap[edge_t].edge_index = torch.Tensor(edge_index).long()
            #if len(snap.edge_index_dict) > 10: break
        snap['node'].x = torch.Tensor([[1] for i in range(max_node)])
        snapshots.append(snap)
        isnap+=1
    return snapshots

class GDELTDurendal(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(GDELTDurendal, self).__init__()
        self.conv1 = DurendalConv(in_channels, hidden_conv_1, metadata)
        #self.update1 = SemanticUpdateWA(n_channels=hidden_conv_1, tau=0.1)
        self.update1 = SemanticUpdateGRU(n_channels=hidden_conv_1)
        #self.update1 = SemanticUpdateMLP(n_channels=hidden_conv_1)
        self.agg1 = SemanticAttention(n_channels=hidden_conv_1)
        
        self.conv2 = DurendalConv(hidden_conv_1, hidden_conv_2, metadata)
        #self.update2 = SemanticUpdateWA(n_channels=hidden_conv_2, tau=0.1)
        self.update2 = SemanticUpdateGRU(n_channels=hidden_conv_2)
        #self.update2 = SemanticUpdateMLP(n_channels=hidden_conv_2)
        self.agg2 = SemanticAttention(n_channels=hidden_conv_2)
        
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
        if snap==0:
            current_dict_1 = out_dict.copy()
        else: 
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()
        out_dict = self.agg1(current_dict_1)
        
        
        out_dict = self.conv2(out_dict, edge_index_dict)
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = self.agg2(current_dict_2)
        
        #HADAMARD MLP
        h = out_dict['node']
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class GDELTATU(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(GDELTATU, self).__init__()
        self.conv1 = DurendalConv(in_channels, hidden_conv_1, metadata)
        self.agg1 = SemanticAttention(n_channels=hidden_conv_1)
        self.update1 = HetNodeUpdateGRU(hidden_conv_1, metadata)
        
        
        self.conv2 = DurendalConv(hidden_conv_1, hidden_conv_2, metadata)
        self.agg2 = SemanticAttention(n_channels=hidden_conv_2)
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
        out_dict = self.agg1(out_dict)
        if snap==0:
            current_dict_1 = out_dict.copy()
        else: 
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()
        
        out_dict = self.conv2(out_dict, edge_index_dict)
        out_dict = self.agg2(out_dict)
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        
        #HADAMARD MLP
        h = out_dict['node']
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class GDELTGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(GDELTGAT, self).__init__()
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
    
class GDELTHAN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_conv_1,
        hidden_conv_2,
        metadata
    ):
        super(GDELTHAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
        self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)
                             
        self.post = Linear(hidden_conv_2, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        out_dict = self.conv2(out_dict, edge_index_dict)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        
        #HADAMARD MLP
        h = out_dict['node']
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

class GDELTGConvGRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_conv_2):
        super(GDELTGConvGRU, self).__init__()
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
    
class GDELTEvolveGCN(torch.nn.Module):
    def __init__(self, in_channels, num_nodes):
        super(GDELTEvolveGCN, self).__init__()
        self.evolve = EvolveGCNH(num_nodes, in_channels)
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
    
class GDELTHEGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes, 
        edge_types
    ):
        super(GDELTHEGCN, self).__init__()
          
        self.conv1 = HeteroConv({edge_t: EvolveGCNH(num_nodes, in_channels) for edge_t in edge_types})
        self.post = torch.nn.Linear(in_channels, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        
        #HADAMARD MLP
        h = out_dict['node']
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)