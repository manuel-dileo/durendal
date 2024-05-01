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

def get_steemit_dataset(preprocess='constant'):
    x = torch.load('steemitth-data/x.pt')
    
    edges = [('node','follow','node'),\
          ('node','comment','node'),\
          ('node','vote','node'),\
          ('node','transaction','node'),\
            ]
    
    snapshots = []
    #for each snap initialize node feature matrix and load the four edge_index
    for isnap in range(0,5):
        snap = HeteroData()
        snap['node'].x = x
        for edge_t in edges:
            edge_index = torch.load(f'steemitth-data/{isnap}_{edge_t}.pt')
            snap[edge_t].edge_index = edge_index.long()
        snapshots.append(snap)
    return snapshots

class SteemitDurendal(torch.nn.Module):
    """
        Durendal model to perform follow link prediction on Steemit following the update-then-aggregate scheme.
    """
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(SteemitDurendal, self).__init__()
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
        
        #past_out_dict is None during test phase and the traning of the first snap
        if past_out_dict_1 is not None: 
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()
            
        out_dict = self.conv1(x_dict, edge_index_dict) #message-passing according to durendal
        if snap==0: #do not perform update step
            current_dict_1 = out_dict.copy()
        else: 
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1) #update
        self.past_out_dict_1 = current_dict_1.copy()
        out_dict = self.agg1(current_dict_1) #then-aggregate
        
        
        out_dict = self.conv2(out_dict, edge_index_dict)
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = self.agg2(current_dict_2)
        
        #HADAMARD MLP as effective decoder for link prediction
        h = out_dict['node']
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class SteemitATU(torch.nn.Module):
    
    """
        Durendal model for follow link prediction on Steemit followin the aggregate-then-update scheme
    """
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(SteemitATU, self).__init__()
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
        out_dict = self.agg1(out_dict) #aggregate
        if snap==0:
            current_dict_1 = out_dict.copy()
        else: 
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1) #then-update
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
    
class SteemitGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(SteemitGAT, self).__init__()
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
    
class SteemitHAN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_conv_1,
        hidden_conv_2,
        metadata
    ):
        super(SteemitHAN, self).__init__()
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

class SteemitGConvGRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_conv_2):
        super(SteemitGConvGRU, self).__init__()
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
    
class SteemitEvolveGCN(torch.nn.Module):
    def __init__(self, in_channels, num_nodes):
        super(SteemitEvolveGCN, self).__init__()
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
    
class SteemitHEGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes, 
        edge_types
    ):
        super(SteemitHEGCN, self).__init__()
          
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