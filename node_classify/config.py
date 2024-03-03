import argparse



parser = argparse.ArgumentParser(description='RHNN.')

parser.add_argument('--dataset', type=str, default='MUTAG', metavar='dataset',

                    help='[AIFB, MUTAG, BGS, AM]')

parser.add_argument('--var', type=str, default='')

parser.add_argument('--task', type=str, default='nc')

parser.add_argument('--model', type=str, default='Hop', help='[GCN, GAT, RGCN, CompGCN, Meta, Hop, GatedGCN]')

parser.add_argument('--seed', type=int, default=1233)

parser.add_argument('--min_epochs', type=int, default=300)

parser.add_argument('--epochs', type=int, default=5000)

parser.add_argument('--patience', type=int, default=100)

parser.add_argument('--cuda', type=int, default=2)

parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument('--dropout', type=float, default=0.6)

parser.add_argument('--one_train', type=int, default=0)

parser.add_argument('--sp_ratio', type=float, default=0.2)



'''node_classify-config'''

parser.add_argument('--drop_e', type=float, default=0.6)

parser.add_argument('--drop_n', type=float, default=0.6)

parser.add_argument('--drop_hid', type=float, default=0.1)

parser.add_argument('--n_in_dim', type=int, default=100)

parser.add_argument('--e_in_dim', type=int, default=100)

parser.add_argument('--t_in_dim', type=int)

parser.add_argument('--n_dim', type=int, default=16)

parser.add_argument('--e_dim', type=int, default=16)

parser.add_argument('--n_layer', type=int, default=2)

parser.add_argument('--e_layer', type=int, default=2)

parser.add_argument('--outputlayer', type=str, default='gcn')

parser.add_argument('--act', type=str, default='tanh')

parser.add_argument('--label_edge_rate', type=float, default=1)  # train_edge_rate

parser.add_argument('--w_edge_reg', type=float, default=0.)  # 0.1

parser.add_argument('--w_node_reg', type=float, default=0.)  # 0.005

parser.add_argument('--beta', type=float, default=0.1)  # weight of type_attr

parser.add_argument('--bias', type=int, default=1)

parser.add_argument('--bn', type=int, default=0)

parser.add_argument('--res', type=int, default=0)

parser.add_argument('--root', type=int, default=1)

parser.add_argument('--heads', type=int, default=8)

parser.add_argument('--atten', type=int, default=0)

parser.add_argument('--num_bases', type=int, default=30)

parser.add_argument('--batch_size', type=int, default=1000000)

parser.add_argument('--opn', dest='opn', default='sub', help='Composition Operation to be used in CompGCN')



'''data-config'''

parser.add_argument('--half_edge', type=str, default='t')

parser.add_argument('--type_attr', type=str, default='t')

parser.add_argument('--dropedge', type=float, default=0.)

parser.add_argument('--dropedge_1', type=float, default=0.)

parser.add_argument('--dropedge_2', type=float, default=0.)



args = parser.parse_args()

