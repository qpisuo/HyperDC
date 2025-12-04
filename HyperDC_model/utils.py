'''
Utilities functions for the framework.
'''
import numpy as np
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from torchmetrics import AveragePrecision
from torchmetrics import Precision, Recall

def parse_args():
    parser = argparse.ArgumentParser()
    
    ##### training hyperparameter #####
    parser.add_argument("--dataset_name", type=str, default='hypergraph_pretrain_noDD', help='dataset name')
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=True, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--gpu", type=int, default=0, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--exp_num", default=1, type=int, help='number of experiments')
    parser.add_argument("--epochs", default=50, type=int, help='number of epochs')
    parser.add_argument("--bs", default=32, type=int, help='batch size')
    parser.add_argument("--train_DG", default="epoch1:1", type=str, help='update ratio in epochs (D updates:G updates)')
    parser.add_argument("--testns", type=str, default='SMCA', help='test negative sampler')
    parser.add_argument("--clip", type=float, default='0.01', help='weight clipping')
    parser.add_argument("--training", type=str, default='wgan', help='loss objective: wgan, none')
    parser.add_argument("--D_lr", default=0.0000005, type=float, help='learning rate')
    parser.add_argument("--G_lr", default=0.00000001, type=float, help='learning rate')
    parser.add_argument("--memory_size", default='128', type=int)
    parser.add_argument("--memory_type", default='sample', type=str)
    
    ##### Discriminator architecture #####
    parser.add_argument("--model", default='hnhn', type=str, help='discriminator: hnhn')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers')   
    parser.add_argument("--alpha_e", default=0, type=float, help='normalization term for hnhn')
    parser.add_argument("--alpha_v", default=0, type=float, help='normalization term for hnhn')
    parser.add_argument("--dim_hidden", default=400, type=int, help='dimension of hidden vector')
    parser.add_argument("--dim_vertex", default=400, type=int, help='dimension of vertex hidden vector')
    parser.add_argument("--dim_edge", default=400, type=int, help='dimension of edge hidden vector')
    
    ##### Generator architecture #####
    parser.add_argument("--gen", type=str, default='MLP', help='generator: MLP')
    
    
    opt = parser.parse_args()
    print(opt.gpu,"device id")
    return opt
    

def gen_size_dist(hyperedges):
    size_dist = {}
    for edge in hyperedges:
        leng = len(edge)
        if leng not in size_dist :
            size_dist[leng] = 0
        size_dist[leng] += 1
    if 1 in size_dist:
        del size_dist[1]
    if 2 in size_dist:   
        del size_dist[2]
    total = sum(v for k, v in size_dist.items())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    return size_dist  
 
def unsqueeze_onehot(onehot):
    edge_size = max(int(onehot.sum().item()), 1)
    onehot_shape = onehot.shape[0]
    unsqueeze = torch.zeros([edge_size, onehot_shape], device=onehot.device)
    nonzero_idx = onehot.nonzero()
    for i, idx in enumerate(nonzero_idx) :
        unsqueeze[i][idx]=1
    return unsqueeze 

def measure(label, pred):
    average_precision = AveragePrecision(task='binary')
    


    auc_roc = metrics.roc_auc_score(np.array(label), np.array(pred))
    ap = average_precision(torch.tensor(pred), torch.tensor(label).long())
    
    
    return auc_roc, ap