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

    print(' '.join(sys.argv[1:]))
    print(f'Using: {args.device}')
    print("Using seed {}.".format(args.seed))

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
        print(f'Num classes: {args.n_classes}')
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
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
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
        if (epoch + 1) % args.log_freq == 0:
            print(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            #val_loss_list.append(val_metrics['loss'].item())
            #val_acc_list.append(val_metrics['acc'].item())
            if (epoch + 1) % args.log_freq == 0:
                print(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    print("Early stopping")
                    break

    '''
    #draw loss
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train_Loss")
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, 'g', label="val_Loss")
    plt.title('loss vs. epoches')
    plt.ylabel('loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, 'r', label="Accuracy")
    plt.plot(np.arange(len(val_acc_list)), val_acc_list, 'g', label="Accuracy")
    plt.xlabel('accuracy vs. epoches')
    plt.ylabel('accuracy')
    plt.show()
    plt.savefig("loss_fig/" + args.dataset + "/" + str(args.seed) + "accuracy_loss.jpg")
    '''

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test', data['adj_train_norm'])
    print(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    print(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if(args.task == 'nc'):
      print("{:.4f}".format(best_test_metrics['acc']))
    else:
      print("{:.4f}".format(best_test_metrics['roc']))
    if args.save:
        os.mknod(os.path.join(save_dir, '{:.4f}test.txt'.format(best_test_metrics['acc'])))
        '''
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        print(f"Saved model in {save_dir}")
        '''

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
