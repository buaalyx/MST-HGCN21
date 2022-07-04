"""Hyperbolic layers."""
import math
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from utils.data_utils import normalize_adj,sparse_mx_to_torch_sparse_tensor
import sys

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att):
        super(HyperbolicGraphConvolution, self).__init__()
        self.bn = nn.BatchNorm1d(in_features, affine=False, track_running_stats=False)
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        if(use_att == 0):
          self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att)
        else:
          self.agg = HypAttAgg(manifold, c_in, out_features, out_features, dropout, use_att)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input       
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)       
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        #init.kaiming_normal_(self.weight, nonlinearity='relu')
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if(str(type(adj)) != '<class \'torch.Tensor\'>'):
          adj = sparse_mx_to_torch_sparse_tensor(adj).to(torch.device('cuda:0'))
        if self.use_att:
            adj_att = self.att(x_tangent, adj)
            support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)

class HypAttAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_dim, out_dim, dropout, use_att):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        #self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        
        self.dropout = dropout
        self.use_att = use_att
        self.reset_parameters()


    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        #nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        
    
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}
        #return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        G = dgl.DGLGraph(adj).to(torch.device('cuda:0'))
        #z = self.fc(x_tangent)
        #G.ndata['z'] = z
        G.ndata['z'] = x_tangent
        # equation (2)
        G.apply_edges(self.edge_attention)
        # equation (3) & (4)
        G.update_all(self.message_func, self.reduce_func)
        output = G.ndata.pop('h')

        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
        #TODO implement of batchnorm

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
