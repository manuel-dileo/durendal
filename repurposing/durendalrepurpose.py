import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell, CrossEntropyLoss
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

from durendal import HetNodeUpdateMLP, HetNodeUpdateGRU

import random

import copy

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv, Linear, HANConv, HGTConv, HeteroConv, GraphConv

import torch
import networkx as nx
import numpy as np

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src')

class DurendalRepurpose(torch.nn.Module):
    """
        Repurposing model for RGCN, HAN and HGT in DURENDAL framework following the aggregate-then-update scheme
    """
    def __init__(self, in_channels, metadata, hidden_conv_1, hidden_conv_2, model='rgcn'):
        
        super(DurendalRepurpose, self).__init__()
        
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        
        edge_types = metadata[1]
     
        if model=='rgcn':
            #RGCN is realized using GraphConv due to no heterogeneity-aware of GCNConv operator
            #This choice was made according to this github thread https://github.com/pyg-team/pytorch_geometric/discussions/3479
            self.conv1 = HeteroConv({edge_t: GraphConv((in_channels[edge_t[0]], in_channels[edge_t[2]]), hidden_conv_1, add_self_loops=False) for edge_t in edge_types}, aggr='sum')
        elif model=='han':
            self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
        else:
            self.conv1 = HGTConv(in_channels, hidden_conv_1, metadata)
        
        #self.update1 = HetNodeUpdateMLP(hidden_conv_1, metadata)
        self.update1 = HetNodeUpdateGRU(hidden_conv_1, metadata)
        
        if model=='rgcn':
            self.conv2 = HeteroConv({edge_t: GraphConv((hidden_conv_1, hidden_conv_1), hidden_conv_2, add_self_loops=False) for edge_t in edge_types}, aggr='sum')
        elif model=='han':
            self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)
        else:
            self.conv2 = HGTConv(hidden_conv_1, hidden_conv_2, metadata)
            
        #self.update2 = HetNodeUpdateMLP(hidden_conv_2, metadata)
        self.update2 = HetNodeUpdateGRU(hidden_conv_2, metadata)
        
        self.post = Linear(hidden_conv_2, 2)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = BCEWithLogitsLoss()
        
        self.past_out_dict_1 = None
        self.past_out_dict_2 = None        
        
    def reset_loss(self,loss=BCEWithLogitsLoss):
        self.loss_fn = loss()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.post.reset_parameters()
        

    def forward(self, x_dict, edge_index_dict, edge_label_index, snap, past_out_dict_1=None, past_out_dict_2=None):
        
        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()
            
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node_t: out.relu() for node_t, out in out_dict.items()}
        if snap==0:
            current_dict_1 = out_dict.copy()
        else: 
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()
        
        out_dict = self.conv2(out_dict, edge_index_dict)
        out_dict = {node_t: out.relu() for node_t, out in out_dict.items()}
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