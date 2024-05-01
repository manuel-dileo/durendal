import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import GDELT, ICEWS18
from torch.nn import RReLU, Flatten

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

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, Linear, HANConv, HeteroConv, SAGEConv
from torch_geometric.nn import ComplEx
from torch.nn import GRUCell, Conv1d

from torch_geometric.nn.inits import glorot


def get_gdelt_dataset(use_random_features=False):
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
            if use_random_features:
                snap['node'].x = torch.randn(max_node,1)
            else:
                snap['node'].x = torch.Tensor([[1] for i in range(max_node)])
            snapshots.append(snap)
            break
    
        if isnap%672!=0: 
            isnap+=1
            continue
        for edge_t, edge_index in edge_index_dict.items():
            snap[edge_t].edge_index = torch.Tensor(edge_index).long()
            #if len(snap.edge_index_dict) > 10: break
        if use_random_features:
            snap['node'].x = torch.randn(max_node,1)
        else:
            snap['node'].x = torch.Tensor([[1] for i in range(max_node)])
        snapshots.append(snap)
        isnap+=1
    return snapshots

def get_icews_dataset(use_random_features=False):
    dataset = ICEWS18(root='ICEWS18/')
    #num_rel = len(set([int(data.rel) for data in dataset]))
    num_rel = 251
    #last_snap = len(t_dict.keys())
    last_snap = 240
    t_dict = {}
    for link in dataset:
        t = int(link.t)
        if t not in t_dict.keys():
            t_dict[t] = []
        t_dict[t].append(link)
    snapshots = []
    max_node = -1
    top20 = ['0', '47', '1', '15', '31', '32', '5', '6', '20', '8', '2', '19', '25', '28', '17', '76', '12', '4', '41', '40'] #top 20 edge types for num of edges, previously computed
    for t, edges in t_dict.items():
        snap = HeteroData()
        edge_index_dict = {('node',r,'node'): [[],[]] for r in top20}
        for edge in edges:
            sub = int(edge.sub)
            rel = str(int(edge.rel))
            obj = int(edge.obj)
            if rel not in top20: continue
            num_node = max(sub, obj)
            if t<60 and num_node > max_node: #60: about 2 months
                max_node = num_node
            if t<60:
                edge_index_dict['node',f'{rel}','node'][0].append(sub)
                edge_index_dict['node',f'{rel}','node'][1].append(obj)
            elif sub > max_node or obj > max_node: pass
            else:
                edge_index_dict['node',f'{rel}','node'][0].append(sub)
                edge_index_dict['node',f'{rel}','node'][1].append(obj)
            
        if t==last_snap-1:
            for edge_t, edge_index in edge_index_dict.items():
                snap[edge_t].edge_index = torch.Tensor(edge_index).long()
            if use_random_features:
                snap['node'].x = torch.randn(max_node,1)
            else:
                snap['node'].x = torch.Tensor([[1] for enc in range(max_node)])
            snapshots.append(snap)
            break
    
        if t%30!=0 or t<60: continue
        for edge_t, edge_index in edge_index_dict.items():
            snap[edge_t].edge_index = torch.Tensor(edge_index).long()
        if use_random_features:
            snap['node'].x = torch.randn(max_node,1)
        else:
            snap['node'].x = torch.Tensor([[1] for enc in range(max_node)])
        snapshots.append(snap)
    return snapshots

def get_steemit_dataset(use_random_features=False):
    x = torch.load('../steemitth/steemitth-data/x.pt')
    if use_random_features:
        nuser = x.size(0)
        x = torch.randn(nuser,1)
    
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
            edge_index = torch.load(f'../steemitth/steemitth-data/{isnap}_{edge_t}.pt')
            snap[edge_t].edge_index = edge_index.long()
        snapshots.append(snap)
    return snapshots

def triple_dot(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return (x * y * z).sum(dim=-1)

class RDurendal(torch.nn.Module):
    """
        Durendal update-then-aggregate for multirelational link prediction task.
    """
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(RDurendal, self).__init__()
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
        
        self.metadata = metadata
        self.rel_emb = torch.nn.Parameter(torch.randn(len(metadata[1]), 2), requires_grad=True)
        self.rel_to_index = {metapath:i for i,metapath in enumerate(metadata[1])}
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.update1.reset_parameters()
        self.agg1.reset_parameters()
        self.conv2.reset_parameters()
        self.update2.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, data, snap, past_out_dict_1=None, past_out_dict_2=None):
        
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
        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)
        
        #ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)
            
            head = out_dict[edge_t[0]][edge_label_index[0]]#embedding src nodes
            head_re_a = head[:,0]
            head_im_a = head[:,1]
            
            tail = out_dict[edge_t[2]][edge_label_index[1]] #embedding dst nodes
            tail_re_a = tail[:,0]
            tail_im_a = tail[:,1]
            
            
            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]
            
            #ComplEx score
            h = torch.Tensor([triple_dot(head_re, rel_re, tail_re) +\
                 triple_dot(head_im, rel_re, tail_im) +\
                 triple_dot(head_re, rel_im, tail_im) -\
                 triple_dot(head_im, rel_im, tail_re) for head_re, head_im, tail_re, tail_im in zip(head_re_a, head_im_a, tail_re_a, tail_im_a)])
            
            h_dict[edge_t] = h
        
        return h_dict, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class RATU(torch.nn.Module):
    """
        Durendal aggregate-then-update for multirelational link prediction task.
    """
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(RATU, self).__init__()
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
        
        self.metadata = metadata
        self.rel_emb = torch.nn.Parameter(torch.randn(len(metadata[1]), 2), requires_grad=True)
        self.rel_to_index = {metapath:i for i,metapath in enumerate(metadata[1])}
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.update1.reset_parameters()
        self.agg1.reset_parameters()
        self.conv2.reset_parameters()
        self.update2.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, data, snap, past_out_dict_1=None, past_out_dict_2=None):
        
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
        
        out_dict = self.conv2(current_dict_1, edge_index_dict)
        out_dict = self.agg2(out_dict)
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = current_dict_2.copy()
        
        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)
        
        #ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)
            
            head = out_dict[edge_t[0]][edge_label_index[0]]#embedding src nodes
            head_re_a = head[:,0]
            head_im_a = head[:,1]
            
            tail = out_dict[edge_t[2]][edge_label_index[1]] #embedding dst nodes
            tail_re_a = tail[:,0]
            tail_im_a = tail[:,1]
            
            
            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]
            
            #ComplEx score
            h = torch.Tensor([triple_dot(head_re, rel_re, tail_re) +\
                 triple_dot(head_im, rel_re, tail_im) +\
                 triple_dot(head_re, rel_im, tail_im) -\
                 triple_dot(head_im, rel_im, tail_re) for head_re, head_im, tail_re, tail_im in zip(head_re_a, head_im_a, tail_re_a, tail_im_a)])
            
            h_dict[edge_t] = h
        
        return h_dict, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

#Below, Temporal Heterogeneous GNN mapped in the DURENDAL framework
class DyHAN(torch.nn.Module):
    """
        DyHAN model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7148053/).
        DyHAN utilizies edge-level and semantic-level attention -> HANConv
        Then, DyHAN leverages temporal-attention to combine node embeddings over time
        It can be reconducted to DURENDAL, following aggregate-then-update schema:
            GNN Encoder: GAT
            Semantic Aggregation: Semantic Attention (HAN)
            Embedding update: Temporal Attention
    """
    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(DyHAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
        self.update1 = HetNodeUpdateTA(hidden_conv_1, metadata)
        
        
        self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)
        self.update2 = HetNodeUpdateTA(hidden_conv_2, metadata)
        
        self.post = Linear(hidden_conv_2, 2)
            
        self.past_out_dict_1 = None
        self.past_out_dict_2 = None
        
        self.loss_fn = BCEWithLogitsLoss()
        
        self.metadata = metadata
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, data, snap, past_out_dict_1=None, past_out_dict_2=None):
        
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
        
        out_dict = self.conv2(current_dict_1, edge_index_dict)
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        
        out_dict = None
        
        #DotProduct followed by LogisticRegression as decoder (following the original paper)
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            
            head = current_dict_2[edge_t[0]][edge_label_index[0]] #embedding src nodes
            tail = current_dict_2[edge_t[2]][edge_label_index[1]] #embedding dst nodes
            
            dot_product = torch.mul(head, tail)
            h = self.post(dot_product)
            h = torch.sigmoid(torch.sum(h.clone(), dim=-1))
            h_dict[edge_t] = h
        
        return h_dict, current_dict_1, current_dict_2
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


class HTGNN(torch.nn.Module):
    """
        HTGNN model (https://dl.acm.org/doi/pdf/10.1145/3583780.3614909).
        HTGNN utilizies edge-level attention -> GAT
                        semantic-level attention -> HAN with attention coefficient following GATv2
                        positional encoding for temporal node embedding
        It can be reconducted to DURENDAL, following aggregate-then-update schema:
            GNN Encoder: GAT
            Semantic Aggregation: Semantic Attention (From GATv2)
            Embedding update: PE (PositionalEncoding) + Temporal Aggregation
    """

    def __init__(
            self,
            in_channels,
            in_channels_int,
            num_nodes,
            metadata: Metadata,
            hidden_conv_1,
            hidden_conv_2,
    ):
        super(HTGNN, self).__init__()
        self.conv1 = DurendalGATConv(in_channels, hidden_conv_1, metadata) #GAT for intra-relation (see original paper)
        self.agg1 = SemanticAttention(n_channels=hidden_conv_1, v2=True) #Semantic attention for inter-relation
        self.update1 = HetNodeUpdatePE(hidden_conv_1, metadata) #PE followed by temporal attention
        self.lin1 = Linear(in_channels_int, hidden_conv_1) #Linear layer for weighted skip-connection
        self.delta1 = torch.nn.Parameter(torch.Tensor([1])) #weight for skip-connection

        self.conv2 = DurendalGATConv(hidden_conv_1, hidden_conv_2, metadata)
        self.agg2 = SemanticAttention(n_channels=hidden_conv_2, v2=True)
        self.update2 = HetNodeUpdatePE(hidden_conv_2, metadata)
        self.lin2 = Linear(hidden_conv_1, hidden_conv_2)
        self.delta2 = torch.nn.Parameter(torch.Tensor([1]))

        self.post = Linear(hidden_conv_2 * 2, 2)

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.agg1.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x_dict, edge_index_dict, data, snap, past_out_dict_1=None, past_out_dict_2=None):

        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()

        x_dict_lin = {node: self.lin1(x) for node,x in x_dict.items()}
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = self.agg1(out_dict)
        if snap == 0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        current_dict_1 = {node: (self.delta1 * current_dict_1[node] + (1-self.delta1) * x_dict_lin[node]) for node in x_dict_lin}
        self.past_out_dict_1 = current_dict_1.copy()

        x_dict_lin = {node: self.lin2(x) for node,x in current_dict_1.items()}
        out_dict = self.conv2(current_dict_1, edge_index_dict)
        out_dict = self.agg2(out_dict)
        if snap == 0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        current_dict_2 = {node: (self.delta2 * current_dict_2[node] + (1-self.delta2) * x_dict_lin[node]) for node in x_dict_lin}
        self.past_out_dict_2 = current_dict_2.copy()

        out_dict = None

        # ConcatMLP as decoder (following the original paper)
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index

            head = current_dict_2[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            tail = current_dict_2[edge_t[2]][edge_label_index[1]]  # embedding dst nodes

            concat = torch.cat((head, tail), dim=1)
            h = self.post(concat)
            h_dict[edge_t] = torch.sum(h, dim=-1)

        return h_dict, current_dict_1, current_dict_2

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

#Below, Temporal Heterogeneous GNN mapped in the DURENDAL framework
class REGCN(torch.nn.Module):
    """
        RE-GCN model (https://dl.acm.org/doi/10.1145/3404835.3462963).
        RE-GCN utilizies a relational GCN as node-encoder -> RGCN
        Then, it leverages a time-gate to combine node embeddings over time.
        Furthermore, it utilizies representations for relations as
            1a) The mean pooling of the node embeddings involved in the relation
            1b) Random inizialized learnable parameters
            1) concatenation between 1a and 1b
            2) Update over time using a GRU unit
        It can be reconducted to DURENDAL, following aggregate-then-update schema:
            GNN Encoder: GCN
            Semantic Aggregation: R-GCN aggregation
            Embedding update: Time-Gate / GRU
    """
    def __init__(
            self,
            in_channels,
            num_nodes,
            metadata: Metadata,
            hidden_conv_1,
            hidden_conv_2,
            output_conv
    ):
        super(REGCN, self).__init__()
        #RGCN using HeteroConv with SAGEConv + mean aggregation
        #As suggested in: https://github.com/pyg-team/pytorch_geometric/discussions/3479
        self.conv1 = HeteroConv({edge_t: SAGEConv((in_channels, in_channels), hidden_conv_1, \
                                                  add_self_loops=False) for edge_t in metadata[1]},
                                aggr='mean')
        self.update1 = HetNodeUpdateGate(hidden_conv_1, metadata)


        self.conv2 = HeteroConv({edge_t: SAGEConv((hidden_conv_1, hidden_conv_1), hidden_conv_2, \
                                                  add_self_loops=False) for edge_t in metadata[1]},
                                aggr='mean')
        self.update2 = HetNodeUpdateGate(hidden_conv_2, metadata)

        self.rel_emb = torch.nn.Parameter(torch.randn(len(metadata[1]), hidden_conv_2))
        self.rel_to_index = {rel:i for i,rel in enumerate(metadata[1])}
        self.update_rel = GRUCell(hidden_conv_2*2, hidden_conv_2)

        self.act = RReLU()
        self.flat = Flatten()

        self.linr = Linear(hidden_conv_2*2, hidden_conv_2)

        self.conv_h = Conv1d(in_channels=hidden_conv_2, out_channels=output_conv, kernel_size=1)
        self.conv_r = Conv1d(in_channels=hidden_conv_2, out_channels=output_conv, kernel_size=1)
        self.conv_t = Conv1d(in_channels=hidden_conv_2, out_channels=output_conv, kernel_size=1)

        self.output_conv = output_conv
        self.post = Linear(output_conv*3, 2)

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None
        self.past_R = None

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.linr.reset_parameters()
        self.conv_h.reset_parameters()
        self.conv_r.reset_parameters()
        self.conv_t.reset_parameters()
        self.post.reset_parameters()

    def forward(self, x_dict, edge_index_dict, data, snap, past_out_dict_1=None, past_out_dict_2=None, past_R=None):

        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()
        if past_R is not None:
            self.past_R = past_R.clone()

        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: self.act(out) for node,out in out_dict.items()}
        if snap==0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()

        out_dict = self.conv2(current_dict_1, edge_index_dict)
        out_dict = {node: self.act(out) for node,out in out_dict.items()}
        if snap==0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = None

        count_types = 0
        for edge_type in self.metadata[1]:
            rel_emb = self.rel_emb[self.rel_to_index[edge_type]].unsqueeze(0)
            dst_type = edge_type[2]
            avg_node = torch.mean(current_dict_2[dst_type][edge_index_dict[edge_type][1]], dim=0).unsqueeze(0)
            r = torch.cat((avg_node, rel_emb),dim=1)
            if count_types == 0:
                R = r.clone()
                count_types+=1
            else:
                R = torch.cat((R,r))
        if snap==0:
            current_R = self.linr(R).clone()
        else:
            current_R = torch.Tensor(self.update_rel(R, self.past_R).detach().numpy())
        self.past_R = current_R.clone()


        #ConvTransE as decoder (following the coriginal paper)
        h_dict = dict()
        flat = self.flat
        out_conv = self.output_conv
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            if edge_label_index[0].size(0) == 0:
                h_dict[edge_t] = torch.Tensor([])
                continue
            head = current_dict_2[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            tail = current_dict_2[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            rel = current_R[self.rel_to_index[edge_t]].unsqueeze(0)
            #CONV
            head_conv = self.conv_h(head.reshape(head.size(1),head.size(0))).reshape(head.size(0),out_conv)
            tail_conv = self.conv_h(tail.reshape(tail.size(1),tail.size(0))).reshape(tail.size(0),out_conv)
            rel_conv = self.conv_r(rel.reshape(rel.size(1),1)).reshape(rel.size(0),out_conv).repeat(head.size(0),1)

            concat = torch.cat((flat(head_conv),flat(rel_conv),flat(tail_conv)),dim=1)
            h = self.post(concat)
            h_dict[edge_t] = torch.sum(h, dim=-1)

        return h_dict, current_dict_1, current_dict_2, current_R

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

#Below, other GNN baselines
    
class RHAN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_conv_1,
        hidden_conv_2,
        metadata
    ):
        super(RHAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
        self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)
                             
        self.post = Linear(hidden_conv_2, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
        self.metadata = metadata
        self.rel_emb = torch.nn.Parameter(torch.randn(len(metadata[1]), 2), requires_grad=True)
        self.rel_to_index = {metapath:i for i,metapath in enumerate(metadata[1])}
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post.reset_parameters()
    
    def forward(self, x_dict, edge_index_dict, data):
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        out_dict = self.conv2(out_dict, edge_index_dict)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        
        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)
        
        #ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)
            
            head = out_dict[edge_t[0]][edge_label_index[0]]#embedding src nodes
            head_re_a = head[:,0]
            head_im_a = head[:,1]
            
            tail = out_dict[edge_t[2]][edge_label_index[1]] #embedding dst nodes
            tail_re_a = tail[:,0]
            tail_im_a = tail[:,1]
            
            
            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]
            
            #ComplEx score
            h = torch.Tensor([triple_dot(head_re, rel_re, tail_re) +\
                 triple_dot(head_im, rel_re, tail_im) +\
                 triple_dot(head_re, rel_im, tail_im) -\
                 triple_dot(head_im, rel_im, tail_re) for head_re, head_im, tail_re, tail_im in zip(head_re_a, head_im_a, tail_re_a, tail_im_a)])
            
            h_dict[edge_t] = h
        
        return h_dict
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class RHEGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes, 
        edge_types
    ):
        super(RHEGCN, self).__init__()
          
        self.conv1 = HeteroConv({edge_t: EvolveGCNH(num_nodes, in_channels) for edge_t in edge_types})
        self.post = torch.nn.Linear(in_channels, 2)
        
        self.edge_types = edge_types
        self.rel_emb = torch.nn.Parameter(torch.randn(len(edge_types), 2), requires_grad=True)
        self.rel_to_index = {metapath:i for i,metapath in enumerate(edge_types)}
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x_dict, edge_index_dict, data):
        
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: out.relu() for node,out in out_dict.items()}
        
        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)
        
        #ComplEx decoder
        h_dict = dict()
        for edge_t in self.edge_types:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)
            
            head = out_dict[edge_t[0]][edge_label_index[0]]#embedding src nodes
            head_re_a = head[:,0]
            head_im_a = head[:,1]
            
            tail = out_dict[edge_t[2]][edge_label_index[1]] #embedding dst nodes
            tail_re_a = tail[:,0]
            tail_im_a = tail[:,1]
            
            
            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]
            
            #ComplEx score
            h = torch.Tensor([triple_dot(head_re, rel_re, tail_re) +\
                 triple_dot(head_im, rel_re, tail_im) +\
                 triple_dot(head_re, rel_im, tail_im) -\
                 triple_dot(head_im, rel_im, tail_re) for head_re, head_im, tail_re, tail_im in zip(head_re_a, head_im_a, tail_re_a, tail_im_a)])
            
            h_dict[edge_t] = h
        
        return h_dict
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
    
#Below, Factorization-method baselines

class ComplEx(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_nodes, 
        metadata,
    ):
        super(ComplEx, self).__init__()
        
        init_size = 1e-3
        self.embedding_dim = embedding_dim
        self.metadata = metadata
        self.node_emb = torch.nn.Parameter(torch.randn((num_nodes, 2*embedding_dim)))
        self.rel_emb = torch.nn.Parameter(torch.randn((len(metadata[1]), 2*embedding_dim)))
        self.rel_to_index = {metapath:i for i,metapath in enumerate(metadata[1])}
        
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, x_dict, edge_index_dict, data):
        
        out_dict = dict()
        
        out_dict['node'] = self.node_emb
        
        #ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)
            
            head = out_dict[edge_t[0]][edge_label_index[0]]#embedding src nodes
            head_re_a = head[:,:self.embedding_dim]
            head_im_a = head[:,self.embedding_dim:]
            
            tail = out_dict[edge_t[2]][edge_label_index[1]] #embedding dst nodes
            tail_re_a = tail[:,:self.embedding_dim]
            tail_im_a = tail[:,self.embedding_dim:]
            
            
            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[:self.embedding_dim]
            rel_im = rel[self.embedding_dim:]
            
            #ComplEx score
            h = torch.Tensor([triple_dot(head_re, rel_re, tail_re) +\
                 triple_dot(head_im, rel_re, tail_im) +\
                 triple_dot(head_re, rel_im, tail_im) -\
                 triple_dot(head_im, rel_im, tail_re) for head_re, head_im, tail_re, tail_im in zip(head_re_a, head_im_a, tail_re_a, tail_im_a)])
            
            h_dict[edge_t] = h
        
        return h_dict
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class TNTComplEx(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_nodes,
        metadata,
        num_timestamps,
        rnn_size
    ):
        super(TNTComplEx, self).__init__()
        
        init_size = 1e-3
        self.embedding_dim = embedding_dim
        self.metadata = metadata
        self.node_emb = torch.nn.Parameter(torch.randn((num_nodes, 2*embedding_dim)))
        self.rel_emb = torch.nn.Parameter(torch.randn((len(metadata[1]), 2*embedding_dim)))
        
        self.rnn_size = rnn_size
        self.num_timestamps = num_timestamps
        self.rnn = torch.nn.GRU(rnn_size, rnn_size)
        self.post_rnn = nn.Linear(rnn_size, 2 * embedding_dim)
        self.h0 = nn.Parameter(torch.randn(1, 1, rnn_size))
        self.rnn_input = nn.Parameter(torch.zeros(self.num_timestamps, 1, rnn_size), requires_grad=False)
        
        self.rel_to_index = {metapath:i for i,metapath in enumerate(metadata[1])}
        
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, x_dict, edge_index_dict, data, ts):
        
        out_dict = dict()
        
        out_dict['node'] = self.node_emb
        
        time_emb, _ = self.rnn(self.rnn_input, self.h0)
        time_emb = torch.squeeze(time_emb)
        time_emb = self.post_rnn(time_emb)
        time_emb_current = time_emb[ts]
        time_re = time_emb_current[:self.embedding_dim]
        time_im = time_emb_current[self.embedding_dim:]
        
        
        #TNTComplEx score
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)
            
            head = out_dict[edge_t[0]][edge_label_index[0]]#embedding src nodes
            head_re_a = head[:,:self.embedding_dim]
            head_im_a = head[:,self.embedding_dim:]
            
            tail = out_dict[edge_t[2]][edge_label_index[1]] #embedding dst nodes
            tail_re_a = tail[:,:self.embedding_dim]
            tail_im_a = tail[:,self.embedding_dim:]
            
            
            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[:self.embedding_dim]
            rel_im = rel[self.embedding_dim:]
            
            #TNTComplEx score
            h = torch.Tensor([torch.sum(((head_re * rel_re * time_re - head_im * rel_im * time_re -
                               head_im * rel_re * time_im - head_re * rel_im * time_im) * tail_re + 
                              (head_im * rel_re * time_re + head_re * rel_im * time_re +
                               head_re * rel_re * time_im - head_im * rel_im * time_im) * tail_im), -1)
                              for head_re, head_im, tail_re, tail_im in zip(head_re_a, head_im_a, tail_re_a, tail_im_a)])
            
            h_dict[edge_t] = h
        
        return h_dict
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)