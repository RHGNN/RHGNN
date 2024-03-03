import os
import os.path as osp
import sys

curPath = osp.abspath(osp.dirname(__file__))
rootPath = osp.split(curPath)[0]
sys.path.append(rootPath)
import logging
from models.nets import GCN, GAT, RGCN, CompGCN, MetaLayer, Hop, SHop, RHop, GatedGCN
from utils.train_eval import init_seeds
from utils.process_graph import build_x, load_data, init_data, type_stats, \
    load_rel_graph
from utils.helper import str2bool
from node_classify.evaluate import ten_train_one_fold, ten_train

Model = {
    'GCN': GCN,
    'GAT': GAT,
    'RGCN': RGCN,
    'CompGCN': CompGCN,
    'Meta': MetaLayer,
    'Hop': Hop,
    'SHop': SHop,
    'RHop': RHop,
    'GatedGCN': GatedGCN
}


def main(args):
    init_seeds(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    '''=========Load data========'''
    dataset = load_data(args.dataset)
    data = dataset[0]
    if args.dataset in ['AM', 'BGS']:
        from utils.process_graph import keep_sub_hop_graph
        keep_sub_hop_graph(data, hop=2)

    rel_data = None
    args.n_classes = dataset.num_classes
    args.n_relations = dataset.num_relations
    args.n_nodes = data.num_nodes
    if args.model in ['Hop', 'SHop', 'RHop', 'Meta', 'TKDE', 'NN', 'GatedGCN', 'RSHNewHop']:
        # '''Init. Node using Degree'''
        # one_hot_degree(data)
        # build_x(data, args.dataset)

        '''Init. Node&Edge&Type'''
        # data.edge_index, data.edge_type = remove_self_loops(data.edge_index, data.edge_type)

        # if str2bool(args.half_edge):
        #     convert_edge_type(data)
        #     half_edge(data)
        #     args.n_relations = args.n_relations // 2
        args.type_attr = str2bool(args.type_attr)
        # train_edge_idx, test_edge_idx = init_data(data, args, dataset.num_relations)
        # args.train_edge_idx = train_edge_idx
        # args.test_edge_idx = test_edge_idx
        init_data(data, args, args.n_relations)
        args.n_in_dim = data.x.size(1)
        args.e_in_dim = data.edge_attr.size(1)
        args.t_in_dim = data.type_attr.size(1)
        data.e_type_w = type_stats(data.edge_type)  # the weight of edge type
        # data.e_type_w = inverse_type_w(data.e_type_w)
        # data.x_type_w = type_stats(torch.cat([data.train_y, data.test_y], dim=0))  # the weight of node type
        # data.x_type_w = inverse_type_w(data.x_type_w)
    elif args.model in ['RSHN', 'RSHNew']:
        build_x(data, args.dataset)
        args.n_in_dim = data.x.size(1)
        rel_data = load_rel_graph(args.dataset)
        args.rel_data_n_features = rel_data.num_features

    '''=====Model and optimizer====='''
    model = Model[args.model](args)
    logging.info(str(model))
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    # logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        data = data.to(args.device)

    '''========Train========'''
    if args.one_train:
        print('one-train')
        # one_train(args, data, model)
        res = ten_train_one_fold(args, data, model, rel_data)
    else:
        print('ten_train')
        res = ten_train(args, data, model, rel_data)

    return res
