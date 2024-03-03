import os
import os.path as osp
import sys

curPath = osp.abspath(osp.dirname(__file__))
rootPath = osp.split(curPath)[0]
sys.path.append(rootPath)
import torch
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from torch_geometric.datasets import Entities
# from torch_geometric.utils import dropout_adj
from type_pretrain.rw import Type2Vec
from utils.process_graph import load_data, convert_edge_type, half_edge, to_undirected
from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes

# AIFB
# 90.83 ± 1.87/90.56 ± 1.94 nohup python type2vec.py --dataset AIFB --patience 1e-3 --half_edge 1 --walk_length 10 --context_size 5 --cuda 3 > ./aifb_type2vec.log 2>&1 &
# 90.28 ± 3.53/90.83 ± 3.72 nohup python type2vec.py --dataset AIFB --patience 1e-3 --half_edge 0 --walk_length 10 --context_size 5 --cuda 3 > ./aifb_type2vec.log 2>&1 &
# 90.56 ± 2.68/90.56 ± 2.68 nohup python type2vec.py --dataset AIFB --patience 1e-3 --half_edge 1 --walk_length 20 --context_size 10 --cuda 3 > ./aifb_type2vec.log 2>&1 &
# 91.94 ± 2.43 nohup python type2vec.py --dataset AIFB --patience 1e-3 --half_edge 0 --walk_length 20 --context_size 10 --cuda 3 > ./aifb_type2vec.log 2>&1 &
#  nohup python type2vec.py --dataset AIFB --patience 1e-4 --half_edge 0 --walk_length 20 --context_size 10 --cuda 3 > ./aifb_type2vec_0_20.log 2>&1 &
#  nohup python type2vec.py --dataset AIFB --patience 1e-4 --half_edge 1 --walk_length 10 --context_size 5 --cuda 3 > ./aifb_type2vec_1_10.log 2>&1 &
#  nohup python type2vec.py --dataset AIFB --patience 1e-4 --half_edge 0 --walk_length 10 --context_size 5 --cuda 3 > ./aifb_type2vec_0_10.log 2>&1 &

# MUTAG
# 69.26 ± 2.72/ nohup python type2vec.py --dataset MUTAG --patience 1e-2 --half_edge 1 --walk_length 10 --context_size 5 --cuda 3 > ./mutag_type2vec.log 2>&1 &
# 68.82 ± 3.91/ nohup python type2vec.py --dataset MUTAG --patience 1e-3 --half_edge 1 --walk_length 10 --context_size 5 --cuda 3 > ./mutag_type2vec.log 2>&1 &
# 67.94 ± 5.58/ nohup python type2vec.py --dataset MUTAG --patience 1e-3 --half_edge 0 --walk_length 10 --context_size 5 --cuda 3 > ./mutag_type2vec.log 2>&1 &
#  nohup python type2vec.py --dataset MUTAG --patience 1e-4 --half_edge 0 --walk_length 20 --context_size 10 --cuda 3 > ./mutag_type2vec.log 2>&1 &
#  nohup python type2vec.py --dataset MUTAG --patience 1e-4 --half_edge 1 --walk_length 20 --context_size 10 --cuda 3 > ./mutag_type2vec_1.log 2>&1 &
#  nohup python type2vec.py --dataset MUTAG --patience 1e-4 --half_edge 0 --walk_length 10 --context_size 5 --cuda 2 > ./mutag_type2vec_0_10.log 2>&1 &
#  nohup python type2vec.py --dataset MUTAG --patience 1e-4 --half_edge 1 --walk_length 10 --context_size 5 --cuda 2 > ./mutag_type2vec_1_10.log 2>&1 &

# nohup python type2vec.py --dataset FB15k-237 --patience 1e-4 --half_edge 0 --walk_length 20 --context_size 10 --cuda 1 > ./FB15k-237_type2vec_0_20.log 2>&1 &
# nohup python type2vec.py --dataset WN18RR --patience 1e-4 --half_edge 0 --walk_length 20 --context_size 10 --cuda 1 > ./WN18RR_type2vec_0_20.log 2>&1 &
# pid 5689 nohup python type2vec.py --dataset BGS --patience 1e-4 --half_edge 0 --walk_length 20 --context_size 10 --cuda 3 > ./bgs_type2vec_0_20.log 2>&1 &
# undone!!!!! nohup python type2vec.py --dataset AM --patience 1e-3 --batch_size 12800 --num_workers 2 --half_edge 0 --walk_length 20 --context_size 10 --cuda 4 > ./am_type2vec_0_20.log 2>&1 &

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='FB15k-237',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM', 'FB15k-237', 'WN18RR'])
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=float, default=1e-3)
parser.add_argument('--walk_length', type=int, default=10)
parser.add_argument('--context_size', type=int, default=5)
parser.add_argument('--walks_per_node', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1028)
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--half_edge', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(1233)
name = args.dataset
if name in ['AIFB', 'MUTAG', 'BGS', 'AM']:
    dataset = load_data(name)
    data = dataset[0]
    if args.half_edge == 1:
        convert_edge_type(data)
        half_edge(data)  # directed graph
        edge_index, edge_type = to_undirected(data.edge_index, data.edge_type, data.num_nodes)
    else:
        edge_index, edge_type = data.edge_index, data.edge_type
else:  # FB15k-237, WN18RR
    edge_index_path = '../link_prediction/data/' + name + '/' + name + '_edge_index'
    edge_type_path = '../link_prediction/data/' + name + '/' + name + '_edge_type'
    edge_type = torch.load(edge_type_path).to('cpu')
    edge_index = torch.load(edge_index_path).to('cpu')
    num_nodes = edge_index.max().item() + 1
    # import scipy.sparse as sparse
    # edge_index = sparse.load_npz(edge_index_path)
    # edge_index, edge_type = to_undirected(edge_index, edge_type, num_nodes)


# '''Randomly drop some edges to avoid segmentation fault'''
# dropedge_rate = 0.3
# data.edge_index, data.edge_type = dropout_adj(data.edge_index, edge_attr=data.edge_type, p=dropedge_rate,
#                                               num_nodes=data.num_nodes)
# data.edge_index, data.edge_type = to_undirected(data.edge_index, data.edge_type, data.num_nodes)


def init_type_mask():
    '''init. edge_type train/test mask'''
    num_types = dataset.num_relations
    idx = torch.randperm(num_types)
    fold = int(num_types * 0.8)
    idx = idx[:fold]
    data.y = torch.arange(0, num_types)

    data.train_mask = torch.zeros(num_types, dtype=torch.bool)
    data.train_mask[idx] = True
    data.test_mask = torch.ones(num_types, dtype=torch.bool)
    data.test_mask[idx] = False


def init_edge_mask():
    '''init. edge_type train/test mask'''
    num_types = dataset.num_relations
    idx = torch.randperm(num_types)
    fold = int(num_types * 0.8)
    idx = idx[:fold]
    data.y = torch.arange(0, num_types)

    data.train_mask = torch.zeros(num_types, dtype=torch.bool)
    data.train_mask[idx] = True
    data.test_mask = torch.ones(num_types, dtype=torch.bool)
    data.test_mask[idx] = False


# init_type_mask()
cuda = args.cuda
if cuda >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
device = 'cuda:' + str(cuda) if int(cuda) >= 0 else 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Type2Vec(edge_index= edge_index, edge_type=edge_type, embedding_dim=args.dim,
                 walk_length=args.walk_length, context_size=args.context_size, walks_per_node=args.walks_per_node,
                 num_negative_samples=1, sparse=True).to(device)

loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    tr_acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.train_mask], data.y[data.train_mask], max_iter=150)
    te_acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return tr_acc, te_acc, z


pre_loss = 100.0
for epoch in range(1, args.epochs):
    loss = train()
    if np.abs(pre_loss - loss) < args.patience:
        break
    pre_loss = loss
    # tr_acc, te_acc, z = test()
    # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {tr_acc:.4f}, Test Acc: {te_acc:.4f}')
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


def save_data_dict(save_file):
    model.eval()
    z = model()
    z = z.cpu()
    torch.save(z, save_file)
    print('save ' + save_file + ' sucessfully!')


save_data_dict(name + '_type_emb_0310_' + str(args.dim) +
               '_halfedge_' + str(args.half_edge) +
               '_walklength' + str(args.walk_length))
# @torch.no_grad()
# def plot_points():
#     colors = [
#         '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
#         '#ffd700'
#     ]
#     model.eval()
#     z = model(torch.arange(data.num_nodes, device=device))
#     z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
#     y = data.y.cpu().numpy()
#
#     plt.figure(figsize=(8, 8))
#     for i in range(dataset.num_classes):
#         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
#     plt.axis('off')
#     plt.show()
# plot_points()
