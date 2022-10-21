import torch
from torch import nn
import dgl
import dgl.function as fn
import numpy as np

class CompGCNCov(nn.Module):
    def __init__(self, in_channels, out_channels, act=None, gcn_drop=0.0, bias=True, opn='corr', device=None):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_str = act  # activation function
        self.device = device
        self.rel = None
        self.opn = opn

        if self.act_str == 'relu':
            self.act = torch.relu
        elif self.act_str == 'tanh':
            self.act = torch.tanh

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])  # 不需要？  逆关系处理
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        # self.drop = nn.Dropout(gcn_drop)
        self.drop = nn.Dropout(0.1)
        # self.bn = torch.nn.BatchNorm1d(out_channels)

        self.bn = nn.BatchNorm1d(out_channels)
        # self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # self.line1 = nn.Linear(out_channels, in_channels)

        self.line_1 = nn.Linear(in_channels*2, out_channels)

        self.line_e_ts = nn.Linear(in_channels*2, in_channels)
        self.line_r_ts = nn.Linear(in_channels*2, in_channels)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(param, gain=nn.init.calculate_gain(self.act_str))
        return param

    def conbine(self, t1, t2):  # 连个tensor融合 三维
        multi_ = t1 * t2
        sub_ = t1 - t2
        add_ = t1 + t2
        results = torch.cat([multi_, sub_, add_], dim=-1)
        return results

    # def message_func(self, edges: dgl.EdgeBatch):
    def message_func(self, edges):
        rel_emd = edges.data['rel_emd']
        edge_num = rel_emd.shape[0]

        # edge_data = self.comp(edges.src['h'], rel_emd)

        time_emd = edges.data['time_emd']
        head_emd = torch.cat([edges.src['h'], time_emd], dim=-1)
        rel_emd = torch.cat([rel_emd, time_emd], dim=-1)
        head_emd = self.line_e_ts(head_emd)
        rel_emd = self.line_r_ts(rel_emd)

        # edge_data = torch.cat([head_emd, rel_emd], dim=-1)
        # edge_data = self.line_1(edge_data)
        edge_data = self.comp(head_emd, rel_emd)



        # rel_emd = rel_emd + time_emd
        # head_emd = edges.src['h'] + time_emd
        # edge_data = self.conbine(head_emd, rel_emd)
        # edge_data = self.line_1(edge_data)

        # edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # edge_data = self.comp(edges.src['h'], rel_emd + time_emd)  # [E, in_channel]
        # edge_data = self.comp(head_emd, rel_emd)  # [E, in_channel]


        # msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
        #                  torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        # msg = msg + edges.data['time_emd']  # E*D

        msg = torch.matmul(edge_data, self.in_w)  # E*D

        # msg = edge_data
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def message_func_train(self, edges):
        msg = self.message_func(edges)['msg']
        mask_index = edges.data['mask_index'].unsqueeze(1)
        # print(mask_index)
        # print(mask_index.size())
        msg = msg * mask_index
        return {'msg': msg}


    def reduce_func_pna(self, nodes):  # PNA的聚合器  还要转换矩阵、转换维度
        h = nodes.mailbox['msg']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        # h = torch.cat([scale(h, D=D, avg_d=self.avg_d_log) for scale in self.scalers], dim=1)
        return {'h': h}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h']) / 2}
        # return {'h': nodes.data['h'] / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
            # return torch.fft.irfft2(com_mult(conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        # g = g.local_var()
        g.ndata['h'] = x
        # g.edata['type'] = edge_type
        # g.edata['norm'] = edge_norm

        edge_type = g.edata['type']  # [E, 1]
        edge_data = rel_repr[edge_type]  # [E, in_channel]
        g.edata['rel_emd'] = edge_data

        # g.update_all(self.message_func, fn.mean(msg='msg', out='h'))  # sum还是mean
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 2
        # x = x / 2

        # if self.bias is not None:
        #     x = x + self.bias



        x = self.bn(x)

        return self.act(x), torch.matmul(rel_repr, self.w_rel)


if __name__ == '__main__':
    compgcn = CompGCNCov(in_channels=10, out_channels=5)
    src, tgt = [0, 1, 0, 3, 2], [1, 3, 3, 4, 4]
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges(src, tgt)  # src -> tgt
    g.add_edges(tgt, src)  # tgt -> src
    edge_type = torch.tensor([0, 0, 0, 1, 1] + [2, 2, 2, 3, 3])
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = in_deg ** -0.5
    norm[np.isinf(norm)] = 0
    g.ndata['xxx'] = norm
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    edge_norm = g.edata.pop('xxx').squeeze()

    x = torch.randn([5, 10])
    rel = torch.randn([4, 10])  # 2*2+1
    x, rel = compgcn(g, x, rel, edge_type, edge_norm)
    print(x.shape, rel.shape)
