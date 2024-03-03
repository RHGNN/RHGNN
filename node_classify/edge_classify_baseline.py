import os
import os.path as osp
import sys

curPath = osp.abspath(osp.dirname(__file__))
rootPath = osp.split(curPath)[0]
sys.path.append(rootPath)
import torch
import logging
from models.nets_ec import GCN, GAT, RGCN, CompGCN, Hop, MetaLayer
from utils.process_graph import convert_edge_type, convert_to_line_graph, load_data, half_edge, init_type
from node_classify.evaluate import cross_folds_ec
from utils.helper import str2bool
from utils.train_eval import init_seeds

Model = {
    'GCN': GCN,
    'GAT': GAT,
    'RGCN': RGCN,
    'CompGCN': CompGCN,
    'Meta': MetaLayer,
    'Hop': Hop
}


def main_ec(args):
    init_seeds(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    '''========Data-line_g========'''
    dataset = load_data(args.dataset)
    data = dataset[0]
    if str2bool(args.half_edge):
        convert_edge_type(data)
        half_edge(data)

    if args.model in ['Hop', 'Meta']:
        g = data
        args.n_nodes = data.num_nodes
        args.n_edges = data.num_edges
        args.n_rels = dataset.num_relations
        args.n_classes = dataset.num_classes
        if args.type_attr:
            data.type_attr = init_type(args.dataset, args.n_rels)
            args.t_in_dim = data.type_attr.size(1)
    else:
        g = convert_to_line_graph(force_directed=False, data=data)
        args.n_nodes = g.num_nodes
        args.n_classes = dataset.num_relations
        args.n_n_classes = dataset.num_classes
        args.n_relations = data.num_nodes

    '''=====Model and optimizer====='''
    model = Model[args.model](args)
    logging.info(str(model))

    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        args.x_label_idx = torch.cat([data.train_idx, data.test_idx], dim=0).to(args.device)
        args.x_label = torch.cat([data.train_y, data.test_y], dim=0).to(args.device)
    else:
        args.x_label_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
        args.x_label = torch.cat([data.train_y, data.test_y], dim=0)

    '''========Train========'''
    res = cross_folds_ec(args, g, model)
    return res