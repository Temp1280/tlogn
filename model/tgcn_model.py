# -*- coding: utf-8 -*-
import torch
from torch import nn
import dgl
from model.layer import CompGCNCov
import torch.nn.functional as F


class TGCN(nn.Module):  # 
    def __init__(self, num_ent, num_rel, input_dim, gcn_dim, n_layer, conv_bias=True, gcn_drop=0.1, opn='mult', act=None, device=None):
        super(TGCN, self).__init__()
        self.act = act  # 
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, -1

        self.init_dim, self.gcn_dim, self.embed_dim = gcn_dim, gcn_dim, gcn_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.n_layer = n_layer

        self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, gcn_drop, conv_bias, opn, device)
        self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, gcn_drop, conv_bias, opn, device) if n_layer == 2 else None
        self.line_time = nn.Linear(input_dim, gcn_dim)

        self.drop = nn.Dropout(gcn_drop)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))  # relu
        # nn.init.xavier_normal_(param, gain=nn.init.calculate_gain(self.act))  # relu
        return param

    def forward(self, g, time_encoder, ent_emds, rel_emds):
        # print('hhh'*100)
        g = g.local_var()

        e_time = g.edata['time']
        # e_time_emd = self.line_time(time_encoder(e_time))
        e_time_emd = time_encoder(e_time)
        g.edata['time_emd'] = e_time_emd

        x, r = ent_emds, rel_emds  # embedding of relations
        # x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
        x, r = self.conv1(g, x, r)
        x = self.drop(x)  # embeddings of entities [num_ent, dim]
        # r = self.drop(r)
        # x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
        x, r = self.conv2(g, x, r) if self.n_layer == 2 else (x, r)
        x = self.drop(x) if self.n_layer == 2 else x
        # r = self.drop(r) if self.n_layer == 2 else r

        return x, r  #
