import os
import os.path as osp
import sys

curPath = osp.abspath(osp.dirname(__file__))
rootPath = osp.split(curPath)[0]
sys.path.append(rootPath)
from node_classify.config import args
from node_classify.main import main
from node_classify.edge_classify_baseline import main_ec

if args.task == 'nc':
    main(args)
    print(args)
elif args.task == 'ec':
    res = main_ec(args)
    desc = '{:.2f} ± {:.2f}, {:.2f} ± {:.2f}'.format(res[0], res[1], res[2], res[3])
    print('Acc and F1 - {}'.format(desc))
