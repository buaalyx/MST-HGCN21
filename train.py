from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data,normalize_adj,sparse_mx_to_torch_sparse_tensor
from utils.train_utils import get_dir_name, format_metrics, adj_gen

import dgl
import networkx as nx
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt
import optuna

def consis_loss(logps, args):
    temp = args.tem
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.consis_rate * loss

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.dataset, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    #print(' '.join(sys.argv[1:]))
    #print(f'Using: {args.device}')
    #print("Using seed {}.".format(args.seed))

    # Load data
    datapath = 'data/' + args.dataset
    data = load_data(args, datapath)
    args.n_nodes, args.feat_dim = data['features'].shape
    
    #spanning tree
    g = dgl.DGLGraph(data['adj_train'])
    g.ndata['f'] = data['features']
    g.ndata['d'] = g.in_degrees()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    g.apply_edges(lambda edges:{'edge_degree':edges.src['d']*edges.dst['d']})
    g.apply_edges(lambda edges:{'similarity':cos(edges.src['f'], edges.dst['f'])})
    g.ndata.pop('d')
    g.ndata.pop('f')
    g.edata['edge_degree'] = 1 / g.edata['edge_degree'].float()
    g_nx = dgl.to_networkx(g, edge_attrs=['edge_degree', 'similarity']).to_undirected()
    T = nx.maximum_spanning_tree(g_nx, weight=args.spt_attr,algorithm=args.spt_alg)
    g_SPT = dgl.from_networkx(T)
    
    #ablation spt or not
    '''

    all_ids_src, all_ids_dst = g_SPT.all_edges()
    all_ids = g_SPT.edge_ids(all_ids_src, all_ids_dst)
    g_SPT.remove_edges(all_ids)
    '''
    
    toAdd_row_list, toAdd_col_list = [], []
    raw_nnz_row_list, raw_nnz_col_list = g.adj(scipy_fmt = 'csr').nonzero()[0].tolist(), g.adj(scipy_fmt = 'csr').nonzero()[1].tolist()
    tree_nnz_row_list, tree_nnz_col_list = g_SPT.adj(scipy_fmt = 'csr').nonzero()[0].tolist(), g_SPT.adj(scipy_fmt = 'csr').nonzero()[1].tolist()
    
    for i in range(len(raw_nnz_row_list)):
        if(T.has_edge(raw_nnz_row_list[i], raw_nnz_col_list[i]) == False):
        #if(g_SPT.has_edges_between(raw_nnz_row_list[i], raw_nnz_col_list[i]) == False):
            toAdd_row_list.append(raw_nnz_row_list[i])
            toAdd_col_list.append(raw_nnz_col_list[i])
    toAdd_list = [toAdd_row_list, toAdd_col_list]    
    adj4dgl_list = adj_gen(args, g_SPT, g, toAdd_list)
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        #print(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    #print(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        nll_loss, log_softmax_p_list = 0., []
        for i in range(len(adj4dgl_list)):
            embeddings = model.encode(data['features'],  adj4dgl_list[i])
            train_metrics = model.compute_metrics(embeddings, data, 'train')
            nll_loss += train_metrics['loss']
            log_softmax_p_list.append(train_metrics['log_softmax_p'])
            
        loss = nll_loss / args.num_adj + consis_loss(log_softmax_p_list, args)
        #train_loss_list.append(loss.item())
        #train_acc_list.append(train_metrics['acc'].item())
        
        loss.backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            #val_loss_list.append(val_metrics['loss'].item())
            #val_acc_list.append(val_metrics['acc'].item())

            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch > args.min_epochs:
                    #print("Early stopping")
                    break

    #print("Optimization Finished!")
    #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test', data['adj_train_norm'])
    #print(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    #print(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if(args.task == 'nc'):
      #print("{:.4f}".format(best_test_metrics['acc']))
      return best_test_metrics['acc']
    else:
      #print("{:.4f}".format(best_test_metrics['roc']))
      return best_test_metrics['roc']


def func_search_cora(trial):
    return {
        "lr": trial.suggest_loguniform("lr", 5e-4, 1e-1),
        "weight_decay": trial.suggest_loguniform("weight_decay", 0.0005, 0.05),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.8),
        "patience": trial.suggest_int("patience", 50, 100, step=10),
        "lr_reduce_freq": trial.suggest_int("lr_reduce_freq", 30, 150, step=15),
        "spt_alg": trial.suggest_categorical("spt_alg", ["kruskal","prim"]),
        "spt_attr": trial.suggest_categorical("spt_attr", ["edge_degree","similarity"]),
        "manifold": trial.suggest_categorical("manifold", ["PoincareBall", "Hyperboloid"]),
        "num_layers": trial.suggest_categorical("num_layers", [3,4,5]),
        "dim": trial.suggest_int("dim", 128, 512, step=16),
        "num_adj": trial.suggest_int("num_adj", 2, 6, step=2),
        "add_rate": trial.suggest_uniform("add_rate", 0.1, 0.9),
        "tem": trial.suggest_uniform("tem", 0.5, 1.5),
        "consis_rate" : trial.suggest_uniform("consis_rate", 0.2, 2),
        "use_att":trial.suggest_categorical("use_att", [0,1]),         
    }
def func_search_pubmed(trial):
    #parameters: {'lr': 0.0021213442368275557, 'weight_decay': 0.0010694050269837725, 'dropout': 0.6048173167494098, 'patience': 105, 'rf': 250, 'dim': 160, 'add-rate': 0.3155011402173754, 'tem': 1.1452812127698557, 'consis-rate': 0.9197993844034479}

    
    return {
        "lr": trial.suggest_uniform("lr", 0.002, 0.003),
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.003),
        "dropout": trial.suggest_uniform("dropout", 0.5, 0.7),
        "patience": trial.suggest_int("patience", 60, 180, step=15),
        "lr_reduce_freq": trial.suggest_int("rf", 200, 300, step=25),
        #"spt-alg": trial.suggest_categorical("spt-alg", ["kruskal","prim"]),
        #"spt-attr": trial.suggest_categorical("spt-attr", ["edge_degree","similarity"]),
        #"manifold": trial.suggest_categorical("manifold", ["PoincareBall", "Hyperboloid"]),
        #"num-layers": trial.suggest_categorical("num-layers", [3]),
        "dim": trial.suggest_int("dim", 96, 192, step=8),
        #"num-adj": trial.suggest_int("num-adj", 2, 6, step=2),
        "add_rate": trial.suggest_uniform("add-rate", 0.3, 0.5),
        "tem": trial.suggest_uniform("tem", 0.8, 1.2),
        "consis_rate" : trial.suggest_uniform("consis_rate", 0.8, 1.2),
    }
    
def func_search_citeseer(trial):
    return {
        "lr": trial.suggest_uniform("lr", 0.0015, 0.0025),
        "weight_decay": trial.suggest_uniform("weight_decay", 0.0005, 0.0015),
        "dropout": trial.suggest_uniform("dropout", 0.4, 0.5),
        "lr_reduce_freq": trial.suggest_int("rf", 170, 200, step=5),
        "patience": trial.suggest_int("patience", 100, 140, step=5),
        "spt_alg": trial.suggest_categorical("spt-alg", ["kruskal", "prim"]),
        "spt_attr": trial.suggest_categorical("spt-attr", ["edge_degree","similarity", "None"]),
        "manifold": trial.suggest_categorical("manifold", ["PoincareBall"]),
        "dim": trial.suggest_int("dim", 320, 400, step=4),
        "num_adj": trial.suggest_int("num-adj", 4, 8, step=2),
        "add_rate": trial.suggest_float("add-rate", 0.01, 0.05, step=0.005),
        "tem": trial.suggest_float("tem", 0.9, 1.2, step= 0.05),
        "consis_rate" : trial.suggest_float("consis-rate", 1.1, 1.5, step=0.1),
        "use-att":trial.suggest_categorical("use-att", [0,1]),   
        "num_layers": trial.suggest_categorical("num-layers", [3]),
    }

def dict_print(dict):
    s = ""
    for key, value in dict.items():
        s += "--" + str(key)
        s += " " + str(value) + " "
    return s

class AutoML(object):
    """
    Args:
        func_search: function to obtain hyper-parameters to search
    """

    def __init__(self, args, n_trials=3, **kwargs):
        self.args = args
        #self.logger = self.args.logger

        #self.seed = kwargs.pop("seed") if "seed" in kwargs else [1]
        assert "func_search" in kwargs
        self.func_search = kwargs["func_search"]
        self.metric = kwargs["metric"] if "metric" in kwargs else None
        self.jobs = kwargs["jobs"]
        self.n_trials = n_trials
        self.best_score = None
        self.best_params = None
        self.default_params = kwargs

    def _objective(self, trials):
        args = self.args
        cur_params = self.func_search(trials)
        print('cur_param: '+str(cur_params))
        for key, value in cur_params.items():
            args.__setattr__(key, value)
        res = []
        for seed in range(10):
            args.__setattr__('seed', seed)
            result = train(args)
            res.append(float(result))
        res = np.array(res)
        score = np.mean(res)
        print("mean:", score)
        print(res)

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_params = cur_params
        return score

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=self.jobs)
        print('best param:'+ dict_print(study.best_params))
        print('best score:'+str(self.best_score))
        #print('best param:', study.best_params)
        return self.best_score



if __name__ == '__main__':
    print('cmd: ' + ' '.join(sys.argv[1:]))
    args = parser.parse_args()
    #train(args)
    funcs = {
        "cora":func_search_cora,
        "pubmed":func_search_pubmed,
        "citeseer": func_search_citeseer
    }
    jobs = {
        "cora":1,
        "pubmed":1,
        "citeseer":1
    }
    func = funcs[args.dataset]
    job = jobs[args.dataset]
    #print(job)
    tool = AutoML(args,  n_trials=job*150, func_search=func, jobs = job)
    result = tool.run()
    print("\nFinal results: ")
    print(result)
    

# --dataset cora --model HGCN --c None --use-att 1
# --dataset pubmed --model HGCN --c None

##citeseer
# python train.py --dataset citeseer  --model HGCN  --num-layers 3  --c None --spt-attr edge_degree
# python train.py --dataset citeseer  --model HGCN  --num-layers 3  --c None --spt-attr similarity
# python train.py --dataset citeseer  --model HGCN  --num-layers 3  --c None --spt-attr None