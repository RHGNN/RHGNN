import sys

import os

curPath = os.path.abspath(os.path.dirname(__file__))

rootPath = os.path.split(curPath)[0]

sys.path.append(rootPath)

import torch.nn.functional as F

from torch.nn import Parameter as Param

from torch.nn import Linear as Lin, ReLU

from torch_scatter import scatter_mean, scatter_add

from models.message_passing import MessagePassing

from torch_sparse import spmm

from torch_geometric.utils import remove_self_loops, add_self_loops, degree, add_remaining_self_loops

from torch_geometric.nn.inits import reset, uniform

from utils.helper import *

from utils.process_graph import add_self_edge_attr_loops, softmax

class EdgeConv_hop(torch.nn.Module):

    def __init__(self,

                 n_in_channels,

                 e_in_channels,

                 out_channels,

                 args,

                 act,

                 bias=False):

        super(EdgeConv_hop, self).__init__()

        self.p = args

        self.n_in_channels = n_in_channels

        self.e_in_channels = e_in_channels

        self.out_channels = out_channels

        self.act = act

        self.drop_hid = torch.nn.Dropout(args.drop_hid)

        self.w_self = Param(torch.Tensor(e_in_channels, out_channels))

        self.w_h = Param(torch.Tensor(n_in_channels, out_channels))

        self.w_t = Param(torch.Tensor(n_in_channels, out_channels))

        self.beta = Param(torch.Tensor(1))

        if bias:

            self.bias = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        if self.p.bn:
            self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.opn == 'cat':
            self.nn = torch.nn.Linear(out_channels + out_channels, out_channels, bias=args.bias)

        self.reset_parameters()

    def reset_parameters(self):

        glorot(self.w_self)

        glorot(self.w_h)

        glorot(self.w_t)

        zeros(self.beta)

        zeros(self.bias)

        if self.p.opn == 'cat':
            self.nn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, edge_type, type_attr=None, label_edge_idx=None, neigh_edge=None):

        row, col = edge_index

        input = edge_attr

        out = torch.mm(edge_attr, self.w_self)

        # 1. scale dimension + dsm

        head = torch.mm(x[row], self.w_h)

        tail = torch.mm(x[col], self.w_t)

        out = out + \
 \
        1 / 2 * self.drop_hid(self.rel_transform(head, out)) + \
 \
        1 / 2 * self.drop_hid(self.rel_transform(tail, out))

    if self.bias is not None: out = out + self.bias

    if self.p.res: out = out + input

    if self.p.bn: out = self.bn(out)

    return self.act(out)


def rel_transform(self, x, e):
    if self.p.opn == 'corr':

        trans_embed = ccorr(x, e)

    elif self.p.opn == 'sub':

        trans_embed = x - e

    elif self.p.opn == 'mult':

        trans_embed = x * e

    elif self.p.opn == 'cat':

        trans_embed = torch.cat([x, e], dim=1)

        trans_embed = self.nn(trans_embed)

    else:

        raise NotImplementedError

    return trans_embed


def reg_edge(self, out, edge_type, type_attr, label_edge_idx):
    num_types = torch.max(edge_type) + 1

    if label_edge_idx is not None:  # node classification

        if type_attr is not None:

            center = type_attr

        else:

            center = scatter_mean(out[label_edge_idx], edge_type[label_edge_idx], dim=0, dim_size=num_types)

        input1 = out[label_edge_idx]

        input2 = F.embedding(edge_type[label_edge_idx], center)

        out[label_edge_idx] = (1 - self.beta) * input1 + self.beta * input2

    else:  # link prediction

        if type_attr is not None:

            center = type_attr

        else:

            center = scatter_mean(out, edge_type, dim=0, dim_size=num_types)

        input1 = out

        input2 = F.embedding(edge_type, center)

        out = (1 - self.beta) * input1 + self.beta * input2

    return out


def __repr__(self):
    return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                               self.out_channels)

class NodeConv_hop(MessagePassing):

    def __init__(self,

                 n_in_channels,

                 e_in_channels,

                 out_channels,

                 args,

                 act):

        super(NodeConv_hop, self).__init__()

        self.atten = args.atten

        self.dropout = args.dropout

        self.p = args

        self.n_in_channels = n_in_channels

        self.e_in_channels = e_in_channels

        self.out_channels = out_channels

        self.drop = torch.nn.Dropout(args.dropout)

        self.drop_hid = torch.nn.Dropout(args.drop_hid)

        self.act = act

        self.w_self = Param(torch.Tensor(n_in_channels, out_channels))

        if self.p.opn == 'cat':

            self.nn_in = torch.nn.Linear(out_channels + e_in_channels, out_channels, bias=args.bias)

            self.nn_out = torch.nn.Linear(out_channels + e_in_channels, out_channels, bias=args.bias)

        else:

            self.nn_in = torch.nn.Linear(e_in_channels, out_channels, bias=args.bias)

            self.nn_out = torch.nn.Linear(e_in_channels, out_channels, bias=args.bias)

        if args.bias:

            self.bias = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        if self.p.bn: self.bn = torch.nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):

        glorot(self.w_self)

        self.nn_in.reset_parameters()

        self.nn_out.reset_parameters()

        # reset(self.nn_in)

        # reset(self.nn_out)

        zeros(self.bias)

        if self.p.atten:
            glorot(self.att)

    @staticmethod
    def norm(edge_index, num_nodes, dtype=None):

        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)

        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr, edge_type, type_attr=None):

        """"""

        if x is None:

            x = self.w_self

            x = self.drop(x)

        else:

            x = torch.mm(x, self.w_self)

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        row, col = edge_index

        # edge_index_inv = torch.stack([col, row], dim=0)

        in_edge, out_edge = row < col, row > col

        in_index, out_index = edge_index[:, in_edge], edge_index[:, out_edge]

        in_type, out_type = edge_type[in_edge], edge_type[out_edge]

        in_edge_attr, out_edge_attr = edge_attr[in_edge], edge_attr[out_edge]

        in_norm = self.norm(in_index, x.size(0), x.dtype)

        out_norm = self.norm(out_index, x.size(0), x.dtype)

        '''out message'''

        out_res = self.propagate('mean', in_index, x=x, edge_attr=in_edge_attr, edge_type=edge_type, norm=in_norm,

                                 mode='in', size=x.size(0))

        '''in message'''

        in_res = self.propagate('mean', out_type, x=x, edge_attr=out_edge_attr, edge_type=edge_type, norm=out_norm,

                                mode='out', size=x.size(0))

        out = x + 1 / 2 * self.drop_hid(in_res) + 1 / 2 * self.drop_hid(out_res)

        return self.act(out)

    def rel_transform(self, x, e):

        if self.p.opn == 'corr':

            trans_embed = ccorr(x, e)

        elif self.p.opn == 'sub':

            trans_embed = x - e

        elif self.p.opn == 'mult':

            trans_embed = x * e

        elif self.p.opn == 'cat':

            trans_embed = torch.cat([x, e], dim=1)

        else:

            raise NotImplementedError

        return trans_embed

    def message(self, edge_index, x_i, x_j, edge_attr, edge_type, norm, mode, size):

        # '''compgcn'''

        # weight = getattr(self, 'w_{}'.format(mode))

        # rel_emb = torch.index_select(rel_embed, 0, edge_type)

        # xj_rel = self.rel_transform(x_j, rel_emb)

        # out = torch.mm(xj_rel, weight)

        # Compute neighbor message

        nn = getattr(self, 'nn_{}'.format(mode))

        if self.p.opn == 'cat':

            xj_edge = self.rel_transform(x_j, edge_attr)

            msg = nn(xj_edge)

        else:

            edge_attr = nn(edge_attr)

            msg = self.rel_transform(x_j, edge_attr)

        if self.atten:

            msg = msg.view(-1, self.heads, self.out_channels // self.heads)

            x_i = x_i.view(-1, self.heads, self.out_channels // self.heads)

            alpha = (torch.cat([x_i, msg], dim=-1) * self.att).sum(dim=-1)

            alpha = F.leaky_relu(alpha, self.negative_slope)

            alpha = softmax(alpha, edge_index[0], size)

            # Sample attention coefficients stochastically.

            alpha = F.dropout(alpha, p=self.dropout, training=True)

            msg = msg * alpha.view(-1, self.heads, 1)

            return msg if norm is None else msg * norm.view(-1, 1, 1)

        else:

            return msg

    def update(self, aggr_out):

        aggr_out = aggr_out.view(-1, self.out_channels)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class NodeConv_hop_wo_half(MessagePassing):

    def __init__(self,

                 n_in_channels,

                 e_in_channels,

                 out_channels,

                 args,

                 act):

        super(NodeConv_hop_wo_half, self).__init__()

        self.atten = args.atten

        self.dropout = args.dropout

        self.p = args

        self.n_in_channels = n_in_channels

        self.e_in_channels = e_in_channels

        self.out_channels = out_channels

        self.drop = torch.nn.Dropout(args.drop_n)

        self.drop_hid = torch.nn.Dropout(args.drop_hid)

        self.act = act

        self.w_self = Param(torch.Tensor(n_in_channels, out_channels))

        if self.p.opn == 'cat':

            self.nn_in = torch.nn.Linear(out_channels + e_in_channels, out_channels, bias=args.bias)

            self.nn_out = torch.nn.Linear(out_channels + e_in_channels, out_channels, bias=args.bias)

        else:

            self.nn_in = torch.nn.Linear(e_in_channels, out_channels, bias=args.bias)

            self.nn_out = torch.nn.Linear(e_in_channels, out_channels, bias=args.bias)

        if args.bias:

            self.bias = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        if self.p.bn: self.bn = torch.nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):

        glorot(self.w_self)

        self.nn_in.reset_parameters()

        self.nn_out.reset_parameters()

        zeros(self.bias)

        if self.p.atten:
            glorot(self.att)

    @staticmethod
    def norm(edge_index, num_nodes, dtype=None):

        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)

        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr, edge_type, type_attr=None):

        """"""

        if x is None:

            x = self.w_self

            x = self.drop(x)

        else:

            x = torch.mm(x, self.w_self)

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        num_edges = edge_index.size(1) // 2

        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]

        in_type, out_type = edge_type[:num_edges], edge_type[num_edges:]

        in_edge_attr, out_edge_attr = edge_attr[:num_edges, :], edge_attr[num_edges:, :]

        '''out message'''

        out_res = self.propagate('mean', in_index, x=x, edge_attr=in_edge_attr, edge_type=in_type,

                                 mode='in', size=x.size(0))

        '''in message'''

        in_res = self.propagate('mean', out_index, x=x, edge_attr=out_edge_attr, edge_type=out_type,

                                mode='out', size=x.size(0))

        out = x + 1 / 2 * self.drop_hid(in_res) + 1 / 2 * self.drop_hid(out_res)

        if self.p.bn: out = self.bn(out)

        return self.act(out)

    def rel_transform(self, x, e):

        if self.p.opn == 'corr':

            trans_embed = ccorr(x, e)

        elif self.p.opn == 'sub':

            trans_embed = x - e

        elif self.p.opn == 'mult':

            trans_embed = x * e

        elif self.p.opn == 'cat':

            trans_embed = torch.cat([x, e], dim=1)

        else:

            raise NotImplementedError

        return trans_embed

    def message(self, edge_index, x_i, x_j, edge_attr, edge_type, mode, size):

        # '''compgcn'''

        # weight = getattr(self, 'w_{}'.format(mode))

        # rel_emb = torch.index_select(rel_embed, 0, edge_type)

        # xj_rel = self.rel_transform(x_j, rel_emb)

        # out = torch.mm(xj_rel, weight)

        # Compute neighbor message

        nn = getattr(self, 'nn_{}'.format(mode))

        if self.p.opn == 'cat':

            xj_edge = self.rel_transform(x_j, edge_attr)

            msg = nn(xj_edge)

        else:

            edge_attr = nn(edge_attr)

            msg = self.rel_transform(x_j, edge_attr)

        return msg

    def update(self, aggr_out):

        aggr_out = aggr_out.view(-1, self.out_channels)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


'''baselines'''


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised

    Classification with Graph Convolutional Networks"

    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::

        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}

        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the

    adjacency matrix with inserted self-loops and

    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:

        in_channels (int): Size of each input sample.

        out_channels (int): Size of each output sample.

        improved (bool, optional): If set to :obj:`True`, the layer computes

            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.

            (default: :obj:`False`)

        cached (bool, optional): If set to :obj:`True`, the layer will cache

            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}

            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the

            cached version for further executions.

            This parameter should only be set to :obj:`True` in transductive

            learning scenarios. (default: :obj:`False`)

        bias (bool, optional): If set to :obj:`False`, the layer will not learn

            an additive bias. (default: :obj:`True`)

        **kwargs (optional): Additional arguments of

            :class:`torch_geometric.nn.conv.MessagePassing`.

    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,

                 bias=True, **kwargs):

        super(GCNConv, self).__init__()

        # aggr='add', **kwargs

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.improved = improved

        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:

            self.bias = Parameter(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        glorot(self.weight)

        zeros(self.bias)

        self.cached_result = None

        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,

             dtype=None):

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,

                                     device=edge_index.device)

        fill_value = 1 if not improved else 2

        edge_index, edge_weight = add_remaining_self_loops(

            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)

        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):

        """"""

        if x is None:

            x = self.weight

        else:

            x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:

            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(

                    'Cached {} number of edges, but found {}. Please '

                    'disable the caching behavior of this layer by removing '

                    'the `cached=True` argument in its constructor.'.format(

                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)

            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,

                                         self.improved, x.dtype)

            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate('add', edge_index=edge_index, x=x, norm=norm)

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True,

                 negative_slope=0.2, dropout=0, bias=True):

        super(GATConv, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels // heads

        self.heads = heads

        self.concat = concat

        self.negative_slope = negative_slope

        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        if bias:

            self.bias = Parameter(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        glorot(self.weight)

        glorot(self.att)

        zeros(self.bias)

    def forward(self, x, edge_index, size=None):

        if x is None:

            x = self.weight

        else:

            x = torch.matmul(x, self.weight)

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)

            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate('add', edge_index=edge_index, x=x, size=x.size(0))

    def message(self, edge_index, x_i, x_j, size):

        edge_index_i = edge_index[0]

        # Compute attention coefficients.

        x_j = x_j.view(-1, self.heads, self.out_channels)

        x_i = x_i.view(-1, self.heads, self.out_channels)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        alpha = softmax(alpha, edge_index_i, size)

        # Sample attention coefficients stochastically.

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_j * alpha.view(-1, self.heads, 1)

        return msg

    def update(self, aggr_out):

        # if self.concat is True:

        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)

        # else:

        #     aggr_out = aggr_out.mean(dim=1)

        aggr_out = aggr_out.view(-1, self.heads * self.out_channels)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):

        return '{}({}, {}, heads={})'.format(self.__class__.__name__,

                                             self.in_channels,

                                             self.out_channels, self.heads)


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling

    Relational Data with Graph Convolutional Networks"

    <https://arxiv.org/abs/1703.06103>`_ paper



    .. math::

        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_0 \cdot \mathbf{x}_i +

        \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}

        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,



    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.

    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which

    stores a relation identifier

    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.



    Args:

        in_channels (int): Size of each input sample.

        out_channels (int): Size of each output sample.

        num_relations (int): Number of relations.

        num_bases (int): Number of bases used for basis-decomposition.

        bias (bool, optional): If set to :obj:`False`, the layer will not learn

            an additive bias. (default: :obj:`True`)

    """

    def __init__(self,

                 in_channels,

                 out_channels,

                 num_relations,

                 num_bases,

                 bias=True,

                 num_nodes=None):

        super(RGCNConv, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.num_relations = num_relations

        self.num_bases = num_bases

        self.num_nodes = num_nodes

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))

        self.att = Param(torch.Tensor(num_relations, num_bases))

        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:

            self.bias = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        size = self.num_bases * self.in_channels

        uniform(size, self.basis)

        uniform(size, self.att)

        uniform(size, self.root)

        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None):

        """"""

        if x is None:

            if self.num_nodes is None:

                x = torch.arange(

                    edge_index.max().item() + 1,

                    dtype=torch.long,

                    device=edge_index.device)

            else:

                x = torch.arange(

                    self.num_nodes,

                    dtype=torch.long,

                    device=edge_index.device)

        return self.propagate(

            'add', edge_index=edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_type, edge_norm):

        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j.dtype == torch.long:

            w = w.view(-1, self.out_channels)

            index = edge_type * self.in_channels + x_j

            out = torch.index_select(w, 0, index)

            return out if edge_norm is None else out * edge_norm.view(-1, 1)

        else:

            w = w.view(self.num_relations, self.in_channels, self.out_channels)

            w = torch.index_select(w, 0, edge_type)

            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):

        if self.root is not None:

            if x.dtype == torch.long:

                aggr_out = aggr_out + self.root

            else:

                aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):

        return '{}({}, {}, num_relations={})'.format(

            self.__class__.__name__, self.in_channels, self.out_channels,

            self.num_relations)


class NodeConv(MessagePassing):

    def __init__(self, in_channels, out_channels, e_dim, act=lambda x: x, params=None):

        super(self.__class__, self).__init__()

        self.p = params

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.act = act

        self.device = None

        if self.p.opn == 'cat':

            self.w_in = get_param((in_channels + e_dim, out_channels))

            self.w_out = get_param((in_channels + e_dim, out_channels))

        else:

            self.w_in = get_param((in_channels, out_channels))

            self.w_out = get_param((in_channels, out_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)

        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.root:

            self.root = get_param((in_channels, out_channels))

        else:

            self.register_parameter('root', None)

        if self.p.bias:

            self.bias = get_param((1, out_channels))

        else:

            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_type, edge_embed):

        if self.device is None:
            self.device = edge_index.device

        # rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        # num_edges = edge_index.size(1) // 2

        num_ent = x.size(0)

        row, col = edge_index

        self.in_index = edge_index

        self.out_index = torch.stack([col, row]).to(self.device)

        self.in_type, self.out_type = edge_type, edge_type

        self.in_norm = self.compute_norm(self.in_index, num_ent)

        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, edge_embed=edge_embed,

                                edge_norm=self.in_norm, mode='in')

        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, edge_embed=edge_embed,

                                 edge_norm=self.out_norm, mode='out')

        out = self.drop(in_res) * (1 / 2) + self.drop(out_res) * (1 / 2)

        if self.root is not None:
            out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        # out = self.bn(out)

        # return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

        return self.act(out)

    def rel_transform(self, ent_embed, edge_embed):

        if self.p.opn == 'corr':

            trans_embed = ccorr(ent_embed, edge_embed)

        elif self.p.opn == 'sub':

            trans_embed = ent_embed - edge_embed

        elif self.p.opn == 'mult':

            trans_embed = ent_embed * edge_embed

        elif self.p.opn == 'cat':

            trans_embed = torch.cat([ent_embed, edge_embed], dim=1)

        else:

            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, edge_embed, edge_norm, mode):

        weight = getattr(self, 'w_{}'.format(mode))

        # edge_emb = torch.index_select(rel_embed, 0, edge_type)

        xj_edge = self.rel_transform(x_j, edge_embed)

        out = torch.mm(xj_edge, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):

        return aggr_out

    def compute_norm(self, edge_index, num_ent):

        row, col = edge_index

        edge_weight = torch.ones_like(row).float()

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges

        deg_inv = deg.pow(-0.5)  # D^{-0.5}

        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):

        return '{}({}, {})'.format(

            self.__class__.__name__, self.in_channels, self.out_channels)


class NodeConvSim(MessagePassing):

    def __init__(self, in_channels, out_channels, e_dim, act=lambda x: x, params=None):

        super(self.__class__, self).__init__()

        self.p = params

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.act = act

        self.device = None

        if self.p.opn == 'cat':

            self.w_in = get_param((in_channels + e_dim, out_channels))

            self.w_out = get_param((in_channels + e_dim, out_channels))

        else:

            self.w_in = get_param((in_channels, out_channels))

            self.w_out = get_param((in_channels, out_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)

        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.root:

            self.root = get_param((in_channels, out_channels))

        else:

            self.register_parameter('root', None)

        if self.p.bias:

            self.bias = get_param((1, out_channels))

        else:

            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_type, edge_embed):

        if self.device is None:
            self.device = edge_index.device

        # rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        # num_edges = edge_index.size(1) // 2

        num_ent = x.size(0)

        row, col = edge_index

        self.in_index = edge_index

        self.out_index = torch.stack([col, row]).to(self.device)

        self.in_type, self.out_type = edge_type, edge_type

        self.in_norm = self.compute_norm(self.in_index, num_ent)

        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, edge_embed=edge_embed,

                                edge_norm=self.in_norm, mode='in')

        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, edge_embed=edge_embed,

                                 edge_norm=self.out_norm, mode='out')

        out = self.drop(in_res) * (1 / 2) + self.drop(out_res) * (1 / 2)

        if self.root is not None:
            out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        out = self.bn(out)

        # return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

        return self.act(out)

    def rel_transform(self, ent_embed, edge_embed):

        if self.p.opn == 'corr':

            trans_embed = ccorr(ent_embed, edge_embed)

        elif self.p.opn == 'sub':

            trans_embed = ent_embed - edge_embed

        elif self.p.opn == 'mult':

            trans_embed = ent_embed * edge_embed

        elif self.p.opn == 'cat':

            trans_embed = torch.cat([ent_embed, edge_embed], dim=1)

        else:

            raise NotImplementedError

        return trans_embed

    def message(self, x_i, x_j, edge_type, edge_embed, edge_norm, mode):

        weight = getattr(self, 'w_{}'.format(mode))

        xj_edge = self.rel_transform(x_i, edge_embed)

        out = torch.mm(xj_edge, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):

        return aggr_out

    def compute_norm(self, edge_index, num_ent):

        row, col = edge_index

        edge_weight = torch.ones_like(row).float()

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges

        deg_inv = deg.pow(-0.5)  # D^{-0.5}

        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):

        return '{}({}, {})'.format(

            self.__class__.__name__, self.in_channels, self.out_channels)


class NNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, in_channels_e, act=lambda x: x, params=None):

        super(NNConv, self).__init__()

        self.p = params

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.act = act

        self.device = None

        self.w_n = Param(torch.Tensor(in_channels, out_channels))

        self.w_e = Param(torch.Tensor(in_channels_e, out_channels))

        if self.p.opn == 'cat':

            self.w = Param(torch.Tensor(out_channels + out_channels, out_channels))

        else:

            self.w = Param(torch.Tensor(out_channels, out_channels))

        if self.p.root:

            self.root = Param(torch.Tensor(in_channels, out_channels))

        else:

            self.register_parameter('root', None)

        if self.p.bias:

            self.bias = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        # self.bn = torch.nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):

        size = self.in_channels

        uniform(size, self.w_n)

        uniform(size, self.w_e)

        uniform(size, self.w)

        uniform(size, self.root)

        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_embed):

        num_ent = x.size(0)

        norm = self.compute_norm(edge_index, num_ent)

        out = self.propagate('add', edge_index, x=x, edge_type=edge_type, edge_embed=edge_embed,

                             edge_norm=norm)

        if self.root is not None:
            out = out + torch.mm(x, self.root)

            # out = out + x

        if self.bias is not None:
            out = out + self.bias

        # out=self.bn(out)

        return self.act(out)

    def rel_transform(self, ent_embed, edge_embed):

        if self.p.opn == 'corr':

            trans_embed = ccorr(ent_embed, edge_embed)

        elif self.p.opn == 'sub':

            trans_embed = ent_embed - edge_embed

        elif self.p.opn == 'mult':

            trans_embed = ent_embed * edge_embed

        elif self.p.opn == 'cat':

            trans_embed = torch.cat([ent_embed, edge_embed], dim=1)

        else:

            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, edge_embed, edge_norm):

        # weight = getattr(self, 'w_{}'.format(mode))

        # edge_emb = torch.index_select(rel_embed, 0, edge_type)

        xj_edge = self.rel_transform(x_j, edge_embed)

        out = torch.mm(xj_edge, self.w)

        # out = xj_edge

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def compute_norm(self, edge_index, num_ent):

        row, col = edge_index

        edge_weight = torch.ones_like(row).float()

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges

        deg_inv = deg.pow(-0.5)  # D^{-0.5}

        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels, n_dim, act=lambda x: x, params=None):

        super(EdgeConv, self).__init__()

        self.p = params

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.w_org = Param(torch.Tensor(in_channels + 2 * n_dim, out_channels))

        self.lin_e = Lin(in_channels, out_channels)

        self.lin_n = Lin(n_dim, out_channels)

        self.act = torch.relu

        self.device = None

        if self.p.bias:

            self.bias = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

    def reset_parameters(self):

        size = self.in_channels

        uniform(size, self.bias)

        self.lin_n.reset_parameters()

        self.lin_e.reset_parameters()

    def forward(self, x, edge_index, edge_attr, edge_type, neb_edge=None):

        row, col = edge_index

        if self.p.opn == 'cat':

            org_edge_attr = torch.cat([x[row], x[col], edge_attr], dim=1)

            out = torch.mm(org_edge_attr, self.w_org)

        elif self.p.opn == 'sub':

            x = self.lin_n(x)

            neb_node = x[row] - x[col]

            out = self.lin_e(edge_attr) + neb_node

        "get the neighbor edge"

        # edge_attr = torch.mm(edge_attr, self.w_neb)

        # num_edges = edge_index.size(1)

        # for i in range(0, num_edges):

        #     neb_edge_attr = torch.mean(edge_attr[neb_edge[i]], dim=0)

        #     edge_attr[i] = org_edge_attr[i] + neb_edge_attr

        '''regularize edge_attr using center'''

        # num_types = torch.max(edge_type) + 1

        # center = scatter_mean(edge_attr, edge_type, dim=0, dim_size=num_types)

        # edge_attr = F.embedding(edge_type, center) + self.beta * edge_attr

        if self.bias is not None:
            out = out + self.bias

        # out = self.drop(out)

        # out = self.bn(out)

        return self.act(out)

        # return out

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class CompGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, params=None):

        super(self.__class__, self).__init__()

        self.p = params

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.num_rels = num_rels

        self.act = act

        self.device = None

        self.w_loop = Param(torch.Tensor(in_channels, out_channels))

        self.w_in = Param(torch.Tensor(in_channels, out_channels))

        self.w_out = Param(torch.Tensor(in_channels, out_channels))

        self.w_rel = Param(torch.Tensor(in_channels, out_channels))

        self.loop_rel = Param(torch.Tensor(1, out_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)

        self.bn = torch.nn.BatchNorm1d(out_channels)

        # if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        self.bias = Param(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_normal_(self.w_loop)

        torch.nn.init.xavier_normal_(self.w_in)

        torch.nn.init.xavier_normal_(self.w_out)

        torch.nn.init.xavier_normal_(self.w_rel)

        torch.nn.init.xavier_normal_(self.loop_rel)

        zeros(self.bias)

    def forward(self, x, edge_index, edge_type, rel_embed):

        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        num_ent = x.size(0)

        row, col = edge_index

        if self.p.task == 'ec':  # half-edge

            self.in_index = torch.stack([row, col], dim=0)

            self.out_index = torch.stack([col, row], dim=0)

            self.in_type, self.out_type = edge_type, edge_type

        else:

            in_edge, out_edge = row < col, row > col

            self.in_index, self.out_index = edge_index[:, in_edge], edge_index[:, out_edge]

            self.in_type, self.out_type = edge_type[in_edge], edge_type[out_edge]

            # num_edges = edge_index.size(1) // 2

            # self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)

        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm(self.in_index, num_ent)

        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,

                                edge_norm=self.in_norm, mode='in')

        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,

                                  edge_norm=None, mode='loop')

        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,

                                 edge_norm=self.out_norm, mode='out')

        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p.bias: out = out + self.bias

        if self.p.bn: out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):

        if self.p.opn == 'corr':

            trans_embed = ccorr(ent_embed, rel_embed)

        elif self.p.opn == 'sub':

            trans_embed = ent_embed - rel_embed

        elif self.p.opn == 'mult':

            trans_embed = ent_embed * rel_embed

        else:

            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):

        weight = getattr(self, 'w_{}'.format(mode))

        rel_emb = torch.index_select(rel_embed, 0, edge_type)

        xj_rel = self.rel_transform(x_j, rel_emb)

        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):

        return aggr_out

    def compute_norm(self, edge_index, num_ent):

        row, col = edge_index

        edge_weight = torch.ones_like(row).float()

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges

        deg_inv = deg.pow(-0.5)  # D^{-0.5}

        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):

        return '{}({}, {}, num_rels={})'.format(

            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


class CompGCNConvBasis(MessagePassing):

    def __init__(self, in_channels, out_channels, num_rels, num_bases, act=lambda x: x, params=None):

        super(self.__class__, self).__init__()

        self.p = params

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.num_rels = num_rels

        self.num_bases = num_bases

        self.act = act

        self.device = None

        self.w_loop = Param(torch.Tensor(in_channels, out_channels))

        self.w_in = Param(torch.Tensor(in_channels, out_channels))

        self.w_out = Param(torch.Tensor(in_channels, out_channels))

        self.w_rel = Param(torch.Tensor(in_channels, out_channels))

        self.loop_rel = Param(torch.Tensor(1, in_channels))

        self.rel_basis = Param(torch.Tensor(self.num_bases, in_channels))

        self.rel_wt = Param(torch.Tensor(self.num_rels, self.num_bases))

        self.drop = torch.nn.Dropout(self.p.dropout)

        self.bn = torch.nn.BatchNorm1d(out_channels)

        self.bias = Param(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_normal_(self.w_loop)

        torch.nn.init.xavier_normal_(self.w_in)

        torch.nn.init.xavier_normal_(self.w_out)

        torch.nn.init.xavier_normal_(self.w_rel)

        torch.nn.init.xavier_normal_(self.loop_rel)

        torch.nn.init.xavier_normal_(self.rel_wt)

        zeros(self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):

        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.mm(self.rel_wt, self.rel_basis)

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        num_ent = x.size(0)

        row, col = edge_index

        if self.p.task == 'ec':  # half-edge

            self.in_index = torch.stack([row, col], dim=0)

            self.out_index = torch.stack([col, row], dim=0)

            self.in_type, self.out_type = edge_type, edge_type

        else:

            in_edge, out_edge = row < col, row > col

            self.in_index, self.out_index = edge_index[:, in_edge], edge_index[:, out_edge]

            self.in_type, self.out_type = edge_type[in_edge], edge_type[out_edge]

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)

        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm(self.in_index, num_ent)

        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,

                                edge_norm=self.in_norm, mode='in')

        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,

                                  edge_norm=None, mode='loop')

        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,

                                 edge_norm=self.out_norm, mode='out')

        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p.bias: out = out + self.bias

        # out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):

        if self.p.opn == 'corr':

            trans_embed = ccorr(ent_embed, rel_embed)

        elif self.p.opn == 'sub':

            trans_embed = ent_embed - rel_embed

        elif self.p.opn == 'mult':

            trans_embed = ent_embed * rel_embed

        else:

            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):

        weight = getattr(self, 'w_{}'.format(mode))

        rel_emb = torch.index_select(rel_embed, 0, edge_type)

        xj_rel = self.rel_transform(x_j, rel_emb)

        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):

        return aggr_out

    def compute_norm(self, edge_index, num_ent):

        row, col = edge_index

        edge_weight = torch.ones_like(row).float()

        deg = scatter_add(edge_weight, row, dim=0,

                          dim_size=num_ent)  # Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]

        deg_inv = deg.pow(-0.5)  # D^{-0.5}

        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):

        return '{}({}, {}, num_rels={})'.format(

            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


class RHopConv(MessagePassing):

    def __init__(self,

                 n_in_channels, e_in_channels, out_channels,

                 act=lambda x: x, params=None, bias=True):

        super(RHopConv, self).__init__()

        '''conv hyper-params'''

        self.atten = 0

        self.n_in_channels = n_in_channels

        self.e_in_channels = e_in_channels

        self.out_channels = out_channels

        self.p = params

        self.act = act

        self.drop = torch.nn.Dropout(self.p.dropout)

        # self.label_idx = self.p.train_edge_idx

        '''attention params'''

        heads = 1

        concat = True

        self.negative_slope = 0.2

        self.concat = concat

        self.heads = heads

        if concat:

            self.out_channels = out_channels // heads

        else:

            self.out_channels = out_channels

        self.att = Param(torch.Tensor(1, heads, 2 * self.out_channels))

        '''conv params'''

        self.w_self_n = Param(torch.Tensor(n_in_channels, heads * self.out_channels))

        self.w_self_e = Param(torch.Tensor(e_in_channels, heads * self.out_channels))

        self.w_node = Param(torch.Tensor(2 * out_channels, out_channels))

        self.w_node_inv = Param(torch.Tensor(2 * out_channels, out_channels))

        self.nn_in = torch.nn.Linear(out_channels + e_in_channels, heads * self.out_channels, bias=True)

        self.nn_out = torch.nn.Linear(out_channels + e_in_channels, heads * self.out_channels, bias=True)

        if bias:

            self.bias_n = Param(torch.Tensor(out_channels))

            self.bias_e = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias_n', None)

            self.register_parameter('bias_e', None)

        # self.beta = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):

        glorot(self.w_self_n)

        glorot(self.w_self_e)

        glorot(self.w_node)

        glorot(self.w_node_inv)

        glorot(self.att)

        self.nn_in.reset_parameters()

        self.nn_out.reset_parameters()

        # self.beta.data.fill_(1)

        zeros(self.bias_n)

        zeros(self.bias_e)

    def forward(self, x, edge_index, edge_attr, type_attr, edge_type):

        '''init.'''

        row, col = edge_index

        if x is None:

            x = self.w_self_n

            x = self.drop(x)

            x0 = x

        else:

            x0 = x

            x = torch.mm(x, self.w_self_n)

        num_nodes = x.size(0)

        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        e0 = edge_attr

        "==========edge-emb=========="

        edge_attr = torch.mm(edge_attr, self.w_self_e)

        endpoint = torch.cat([x[row], x[col]], dim=1)

        endpoint = torch.mm(endpoint, self.w_node)

        endpoint_inv = torch.cat([x[col], x[row]], dim=1)

        endpoint_inv = torch.mm(endpoint_inv, self.w_node_inv)

        edge_attr = 1 / 2 * self.drop(endpoint) + 1 / 2 * self.drop(endpoint_inv) + edge_attr

        "==========node-emb========="

        in_norm = 1. / degree(col, num_nodes)[col]  # Norm by in-degree.

        out_norm = 1. / degree(row, num_nodes)[row]  # Norm by out-degree.

        '''out message'''

        out_res = self.propagate('add', edge_index, x=x, edge_attr=edge_attr, edge_type=edge_type, edge_norm=out_norm,

                                 mode='out', size=x.size(0))

        '''in message'''

        edge_index_inv = torch.stack([col, row], dim=0)

        in_res = self.propagate('add', edge_index_inv, x=x, edge_attr=edge_attr, edge_type=edge_type, edge_norm=in_norm,

                                mode='in', size=x.size(0))

        node_emb = 1 / 2 * self.drop(in_res) + 1 / 2 * self.drop(out_res) + x

        if self.bias_n is not None:
            node_emb = node_emb + self.bias_n

        if self.p.bn:
            node_emb = self.norm(node_emb)

        node_emb = self.act(node_emb)

        edge_attr = self.act(edge_attr)

        return node_emb, edge_attr

    def message(self, edge_index, x_i, x_j, edge_attr, edge_type, edge_norm, mode, size):

        nn = getattr(self, 'nn_{}'.format(mode))

        xj_edge = torch.cat([x_j, edge_attr], dim=1)

        msg = nn(xj_edge)

        msg = msg.view(-1, self.heads, self.out_channels)

        if self.atten:

            x_i = x_i.view(-1, self.heads, self.out_channels)

            alpha = (torch.cat([x_i, msg], dim=-1) * self.att).sum(dim=-1)

            alpha = F.leaky_relu(alpha, self.negative_slope)

            alpha = softmax(alpha, edge_index[0], size)

            # Sample attention coefficients stochastically.

            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            msg = msg * alpha.view(-1, self.heads, 1)

        return msg if edge_norm is None else msg * edge_norm.view(-1, 1, 1)

    def update(self, aggr_out):

        if self.concat:

            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)

        else:

            aggr_out = aggr_out.mean(dim=1)

        return aggr_out

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class GatedConv_n(MessagePassing):

    def __init__(self,

                 n_in_channels, e_in_channels, out_channels,

                 params=None, bias=True):
        super(GatedConv_n, self).__init__()

        self.in_channels = n_in_channels

        self.out_channels = out_channels

        self.p = params

        self.act = torch.relu

        self.u = Param(torch.Tensor(n_in_channels, out_channels))

        self.v = Param(torch.Tensor(n_in_channels, out_channels))

        self.bn = torch.nn.BatchNorm1d(out_channels)

        self.drop = torch.nn.Dropout(self.p.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.u)

        glorot(self.v)

    def forward(self, x, edge_attr, edge_index, edge_type):
        """"""

        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        "==========node-emb========="

        xj = torch.mm(x, self.v)

        out_res = self.propagate('mean', edge_index, x=xj, edge_attr=edge_attr)

        node_emb = torch.mm(x, self.u) + out_res

        node_emb = self.act(node_emb)

        return node_emb

    def message(self, x_j, edge_attr):
        xj_edge = x_j * edge_attr

        return xj_edge

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class GatedConv_e(torch.nn.Module):

    def __init__(self,

                 n_in_channels, e_in_channels, out_channels,

                 params=None, bias=True):
        super(GatedConv_e, self).__init__()

        self.in_channels = n_in_channels

        self.out_channels = out_channels

        self.p = params

        self.act = torch.relu

        self.A = Param(torch.Tensor(n_in_channels, out_channels))

        self.B = Param(torch.Tensor(n_in_channels, out_channels))

        self.C = Param(torch.Tensor(e_in_channels, out_channels))

        self.bn = torch.nn.BatchNorm1d(out_channels)

        self.drop = torch.nn.Dropout(self.p.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.A)

        glorot(self.B)

        glorot(self.C)

    def forward(self, x, edge_attr, edge_index, edge_type):
        """"""

        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        row, col = edge_index

        "==========edge-emb=========="

        h_i = torch.mm(x, self.A)

        h_j = torch.mm(x, self.B)

        edge_attr = h_i[row] + h_j[col] + torch.mm(edge_attr, self.C)

        edge_attr = self.act(edge_attr)

        return edge_attr

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class RelationConv(torch.nn.Module):

    def __init__(self, eps=0, train_eps=False, requires_grad=True):

        super(RelationConv, self).__init__()

        self.initial_eps = eps

        if train_eps:

            self.eps = torch.nn.Parameter(torch.Tensor([eps]))

        else:

            self.register_buffer('eps', torch.Tensor([eps]))

        '''beta'''

        self.requires_grad = requires_grad

        if requires_grad:

            self.beta = Param(torch.Tensor(1))

        else:

            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):

        self.eps.data.fill_(self.initial_eps)

        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x, edge_index, edge_attr):

        """"""

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        row, col = edge_index

        '''co-occurrence rate'''

        for i in range(len(x)):
            mask = torch.eq(row, i)

            edge_attr[mask] = F.normalize(edge_attr[mask], p=2, dim=0)

        '''add-self-loops'''

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index

        edge_attr = add_self_edge_attr_loops(edge_attr, x.size(0))

        x = F.normalize(x, p=2, dim=-1)

        beta = self.beta if self.requires_grad else self._buffers['beta']

        alpha = beta * edge_attr

        alpha = softmax(alpha, row, num_nodes=x.size(0))

        '''Perform the propagation.'''

        out = spmm(edge_index, alpha, x.size(0), x.size(1), x)

        out = (1 + self.eps) * x + out

        return out

    def __repr__(self):

        return '{}()'.format(self.__class__.__name__)


class SNodeConv(MessagePassing):

    def __init__(self,

                 in_channels,

                 out_channels,

                 nn,

                 aggr="add",

                 root_weight=False,

                 bias=False):

        super(SNodeConv, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.nn = nn

        self.aggr = aggr

        self.weight = Param(torch.Tensor(in_channels, out_channels))

        if root_weight:

            self.root = Param(torch.Tensor(in_channels, out_channels))

        else:

            self.register_parameter('root', None)

        if bias:

            self.bias = Param(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        reset(self.nn)

        size = self.in_channels

        uniform(size, self.weight)

        uniform(size, self.root)

        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_weight = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        edge_weight = self.nn(edge_weight).view(-1, self.out_channels)

        x = torch.matmul(x, self.weight)

        return self.propagate(self.aggr, edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):

        message = x_j - edge_weight

        return message

    def update(self, aggr_out, x):

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out + x

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


'''Hop for ec'''


class NodeConv_ec(MessagePassing):

    def __init__(self,

                 in_channels,

                 out_channels,

                 nn,

                 aggr,

                 root_weight=False,

                 bias=False):

        super(NodeConv_ec, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.nn = nn

        self.aggr = aggr

        self.beta = Parameter(torch.Tensor(1))

        if root_weight:

            self.root = Parameter(torch.Tensor(in_channels, out_channels))

        else:

            self.register_parameter('root', None)

        if bias:

            self.bias = Parameter(torch.Tensor(out_channels))

        else:

            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        reset(self.nn)

        # for item in [self.nn]:

        #     if hasattr(item, 'reset_parameters'):

        #         item.reset_parameters()

        size = self.in_channels

        uniform(size, self.root)

        uniform(size, self.bias)

        self.beta.data.fill_(1)

    def forward(self, x, edge_index, edge_attr, type_attr, edge_norm, edge_type):

        """"""

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        row, col = edge_index

        '''regularize edge_attr using center'''

        # num_types = torch.max(edge_type) + 1

        # center = scatter_mean(edge_attr, edge_type, dim=0, dim_size=num_types)

        # edge_attr = F.embedding(edge_type, center) + self.beta * edge_attr

        # if type_attr is not None:

        #     type_attr = type_attr.unsqueeze(-1) if type_attr.dim() == 1 else type_attr

        #     # out = torch.cat([edge_attr, type_attr], dim=1)

        #     edge_attr = torch.max(edge_attr, type_attr)

        out = torch.cat([x[col], edge_attr], dim=1)

        # out = self.nn(out)

        "another edge"

        out_des = torch.cat([x[row], edge_attr], dim=1)

        # out_des = self.nn(out_des)

        if edge_norm is not None:
            out = out * edge_norm.view(-1, 1)

            out_des = out_des * edge_norm.view(-1, 1)

        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        out_des = scatter_add(out_des, col, dim=0, dim_size=x.size(0))

        out = out + out_des

        if self.aggr == 'mean':
            out = out / 2

        out = self.nn(out)

        if self.root is not None:
            out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j):

        return x_j

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)


class EdgeConv_Mask(torch.nn.Module):

    def __init__(self,

                 in_channels,

                 out_channels,

                 nn):
        super(EdgeConv_Mask, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.nn = nn

        self.beta = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

        self.beta.data.fill_(1)

    def forward(self, x, edge_index, edge_attr, edge_type):
        row, col = edge_index

        # if type_attr is None:

        #     edge_attr = edge_attr

        # else:

        #     edge_attr = torch.max(edge_attr, type_attr)

        edge_attr = torch.cat([x[row], x[col], edge_attr], dim=1)

        edge_attr = self.nn(edge_attr)

        return edge_attr

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,

                                   self.out_channels)
