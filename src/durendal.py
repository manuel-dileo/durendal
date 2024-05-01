import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
import random
import gc
import copy
from itertools import permutations

from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear, GraphConv, GATConv

import networkx as nx
import numpy as np

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax

import math

from sklearn.metrics import *

class DurendalConv(MessagePassing):
    """
        A class that perform the message-passing operation according to the DURENDAL architecture.
        In DurendalConv, messages are exchanged between each edge type and
        partial node representations for each relation type are computed for each node type.
    """
    
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr=None, node_dim=0, **kwargs)
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.dropout = dropout
        
        #self.proj = nn.ModuleDict()
        #for node_type, in_channels in self.in_channels.items():
            #self.proj[node_type] = Linear(in_channels, out_channels)
        
        #A message-passing layer for each relation type
        self.conv = nn.ModuleDict()
        for edge_type in metadata[1]:
            src, _, dst = edge_type
            edge_type = '__'.join(edge_type)
            self.conv[edge_type] = GraphConv((in_channels[src],in_channels[dst]), out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        #reset(self.proj)
        reset(self.conv)
    
    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ):
        x_node_dict, out_dict = {}, {}

        # Iterate over node types to linear project the node features in the same space:
        for node_type, x in x_dict.items():
            #x_node_dict[node_type] = self.proj[node_type](x)
            x_node_dict[node_type] = x_dict[node_type]
            out_dict[node_type] = {}
    
        #Iterate over edge types to perform convolution:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, r_type, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            x = (x_src, x_dst)
            out = self.conv[edge_type](x, edge_index)
            
            out = F.relu(out)
            out_dict[dst_type][r_type] = out
        
        #Retrieve the node representations even if they have no in-edges (filtered out by default by PyG)
        for node_type, out in out_dict.items():
            if out=={}:
                for edge_type in edge_index_dict.keys():
                    src_type, r_type, _ = edge_type
                    if src_type == node_type:
                        out_dict[node_type][r_type] = x_node_dict[node_type]
        
        return out_dict

class DurendalGATConv(MessagePassing):
    """
        A class that perform the message-passing operation according to the DURENDAL architecture.
        In DurendalConv, messages are exchanged between each edge type and
        partial node representations for each relation type are computed for each node type.
    """

    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            metadata: Metadata,
            dropout: float = 0.0,
            **kwargs,
    ):
        super().__init__(aggr=None, node_dim=0, **kwargs)
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.dropout = dropout

        #self.proj = nn.ModuleDict()
        #for node_type, in_channels in self.in_channels.items():
        #self.proj[node_type] = Linear(in_channels, out_channels)

        #A message-passing layer for each relation type
        self.conv = nn.ModuleDict()
        for edge_type in metadata[1]:
            src, _, dst = edge_type
            edge_type = '__'.join(edge_type)
            self.conv[edge_type] = GATv2Conv((in_channels[src],in_channels[dst]), out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        #reset(self.proj)
        reset(self.conv)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Adj]
    ):
        x_node_dict, out_dict = {}, {}

        # Iterate over node types to linear project the node features in the same space:
        for node_type, x in x_dict.items():
            #x_node_dict[node_type] = self.proj[node_type](x)
            x_node_dict[node_type] = x_dict[node_type]
            out_dict[node_type] = {}

        #Iterate over edge types to perform convolution:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, r_type, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            x = (x_src, x_dst)
            out = self.conv[edge_type](x, edge_index)

            out = F.relu(out)
            out_dict[dst_type][r_type] = out

        #Retrieve the node representations even if they have no in-edges (filtered out by default by PyG)
        for node_type, out in out_dict.items():
            if out=={}:
                for edge_type in edge_index_dict.keys():
                    src_type, r_type, _ = edge_type
                    if src_type == node_type:
                        out_dict[node_type][r_type] = x_node_dict[node_type]

        return out_dict
    
#Below, update modules for Update-Then-Aggregate schema:
    
class SemanticUpdateGRU(torch.nn.Module):
    """
        Update the partial representation of nodes using GRU Cell.
    """
    def __init__(
        self,
        n_channels,
    ):
        super(SemanticUpdateGRU, self).__init__()
        self.updater = GRUCell(n_channels, n_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.updater.reset_parameters()
    
    def forward(self, current_in_dict, past_in_dict=None):
        if past_in_dict is None: return current_in_dict
        out_dict = {}
        #For each node type, for each relation type, update the node states.
        for node_type, current_in in current_in_dict.items():
            out_dict[node_type] = {}
            for r_type, current_emb in current_in.items():
                past_emb = past_in_dict[node_type][r_type]
                out = torch.Tensor(self.updater(current_emb.clone(), past_emb.clone()).detach().numpy())
                out_dict[node_type][r_type] = out
        return out_dict
    
class SemanticUpdateMLP(torch.nn.Module):
    """
        Update the partial representation of nodes using ConcatMLP.
    """
    def __init__(
        self,
        n_channels,
    ):
        super(SemanticUpdateMLP, self).__init__()
        self.updater = Linear(n_channels*2, n_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.updater.reset_parameters()
    
    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            out_dict[node_type] = {}
            for r_type, current_emb in current_in.items():
                past_emb = past_in_dict[node_type][r_type]
                hin = torch.cat((current_emb.clone(),past_emb.clone()), dim=1)
                out = torch.Tensor(self.updater(hin).detach().numpy())
                out_dict[node_type][r_type] = out
        return out_dict
    
class SemanticUpdateWA(torch.nn.Module):
    """
        Update the partial representation of nodes using a weighted average.
    """
    def __init__(
        self,
        n_channels,
        tau = 0.20 #weight to past information
    ):
        super(SemanticUpdateWA, self).__init__()
        self.tau = tau
    
    def reset_parameters(self): pass
    
    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            out_dict[node_type] = {}
            for r_type, current_emb in current_in.items():
                past_emb = past_in_dict[node_type][r_type]
                out = torch.Tensor((self.tau * past_emb.clone() + (1-self.tau) * current_emb.clone()).detach().numpy())
                out_dict[node_type][r_type] = out
        return out_dict
    
#Below, update modules for aggregate-then-update schema
    
class HetNodeUpdateMLP(torch.nn.Module):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes using ConcatMLP.
    (aggregate-then-update paradigm)
    """
    def __init__(
        self,
        in_channels,
        metadata
    ):
        super(HetNodeUpdateMLP, self).__init__()
        
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
            
        self.in_channels = in_channels
        self.metadata = metadata
        
        
        self.update = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.update[node_type] = Linear(in_channels*2, in_channels)
    
    def reset_parameters(self):
        for layer in self.update.values():
            layer.reset_parameters()
    
    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            update_in = torch.cat((current_in.clone(), past_in.clone()),dim=1)
            out_dict[node_type] = torch.Tensor(self.update[node_type](update_in).detach().numpy())
        return out_dict
    
class HetNodeUpdateGRU(torch.nn.Module):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes using ConcatMLP.
    (aggregate-then-update paradigm)
    """
    def __init__(
        self,
        in_channels,
        metadata
    ):
        super(HetNodeUpdateGRU, self).__init__()
        
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
            
        self.in_channels = in_channels
        self.metadata = metadata
        
        
        self.update = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.update[node_type] = GRUCell(in_channels, in_channels)
    
    def reset_parameters(self):
        for layer in self.update.values():
            layer.reset_parameters()
    
    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            out_dict[node_type] = torch.Tensor(self.update[node_type](current_in, past_in).detach().numpy())
        return out_dict
    
class HetNodeUpdateTA(MessagePassing):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes using temporal attention (DyHan [Yang et al., 2020])
    (aggregate-then-update paradigm)
    Specifically, DyHan utilizes Scaled Dot-Product Attention
    """
    def __init__(
        self,
        in_channels,
        metadata,
        **kwargs
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
            
        self.in_channels = in_channels
        self.metadata = metadata
        
        self.k_lin = Linear(in_channels, in_channels, bias=False)
        self.q_lin = Linear(in_channels, in_channels, bias=False)
        self.v_lin = Linear(in_channels, in_channels, bias=False)
    
    def reset_parameters(self):
        self.k_lin.reset_parameters()
        self.q_lin.reset_parameters()
        self.v_lin.reset_parameters()
        
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        query = None
        key = None
        attn_weight += attn_bias
        attn_bias = None
        gc.collect()
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return (attn_weight @ value)
    
    def group_by_temporal_attention(
        self,
        xs: List[Tensor]
    ):
        out = xs[-1]
        Q = self.q_lin(out)
        K = self.k_lin(out)
        V = self.v_lin(out)
        return self.scaled_dot_product_attention(Q, K, V)
    
    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            xs = []
            past_in = past_in_dict[node_type]
            xs.append(past_in)
            xs.append(current_in)
            out_dict[node_type] = self.group_by_temporal_attention(xs)
            xs = None
            gc.collect()
        return out_dict
    
class HetNodeUpdatePE(torch.nn.Module):
    """
    Implementation of a temporal update embedding module using positional encoding.
    Temporal update embedding module as defined by HTGNN (Fan et al, 2021)
    (aggregate-then-update paradigm)
    """
    def __init__(
        self,
        in_channels,
        metadata
    ):
        super(HetNodeUpdatePE, self).__init__()
            
        self.in_channels = in_channels
        self.metadata = metadata
        
        self.q_lin = Linear(in_channels, in_channels)
        self.k_lin = Linear(in_channels, in_channels)
        self.v_lin = Linear(in_channels, in_channels)
    
    def reset_parameters(self):
        self.q_lin.reset_parameters()
        self.k_lin.reset_parameters()
        self.v_lin.reset_parameters()
        

    def positional_encoding(self, x, t):
        n = x.size(1)
        indices = torch.arange(n)
        encoding = torch.where(indices % 2 == 0, torch.sin(t / (1000 ** (2 * indices / n))), torch.cos(t / (1000 ** (2 * indices / n))))
        result = x + encoding
        return result
        
    
    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            current_pe = self.positional_encoding(current_in,1)
            past_pe = self.positional_encoding(past_in,0)
            q = self.q_lin(current_pe)
            k = self.k_lin(past_pe)
            gamma = q*k
            v = self.v_lin(current_pe)
            out_dict[node_type] = F.leaky_relu(gamma*v)
        return out_dict

class HetNodeUpdateGate(MessagePassing):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes
    using the temporal gate of RE-GCN (https://dl.acm.org/doi/10.1145/3404835.3462963)
    (aggregate-then-update paradigm)
    """
    def __init__(
            self,
            in_channels,
            metadata,
            **kwargs
    ):
        super(HetNodeUpdateGate, self).__init__()

        self.in_channels = in_channels
        self.metadata = metadata

        self.u_lin = Linear(in_channels, in_channels)

    def reset_parameters(self):
        self.u_lin.reset_parameters()

    def temporal_gate(self, current, past):
        u = torch.sigmoid(self.u_lin(past).sum(dim=1)).unsqueeze(dim=1)
        out = (u * current) + ((1 - u) * past)
        return out

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            out_dict[node_type] = self.temporal_gate(current_in, past_in)
            gc.collect()
        return out_dict
    
#Below, aggregation functions at semantic level:
class SemanticAttention(MessagePassing):
    """
        Aggregation scheme for partial node representations using semantic-level attention mechanism,
        as described in "Heterogeneous Graph Attention Network" (Wang et al., 2020)
    """
    
    def __init__(
        self,
        n_channels: int,
        v2: bool = False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.k_lin = nn.Linear(n_channels, n_channels)
        self.q = nn.Parameter(torch.Tensor(1, n_channels))
        self.v2 = v2
        self.reset_parameters()
    
    def reset_parameters(self):
        self.k_lin.reset_parameters()
        glorot(self.q)
    
    def group_by_semantic_attention(
        self,
        xs: List[Tensor],
        q: nn.Parameter,
        k_lin: nn.Module,
    ):
        if len(xs) == 0:
            return None
        else:
            num_edge_types = len(xs)
            out = torch.stack(xs)
            if out.numel() == 0:
                return out.view(0, out.size(-1)), None
            if self.v2:
                attn_score = (q * (torch.tanh(k_lin(out)).mean(1))).sum(-1) #see HTGNN by Fan et al.
            else:
                attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1) #see the Heterogeneous Graph Attention Network by Wang et al.
            attn = F.softmax(attn_score, dim=0)
            out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out
    
    def forward(self, in_dict):
        out_dict = {}
        for node_type, partial_in in in_dict.items():
            xs = []
            for _,v in partial_in.items():
                xs.append(v)
            out = self.group_by_semantic_attention(xs, self.q, self.k_lin)
            out_dict[node_type] = out
        return out_dict