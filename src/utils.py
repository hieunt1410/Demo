import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_tensor(graph):
    """Convert to sparse tensor"""
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    
    return graph

def split_batch_item(items, pop):
    g1, g2 = [], []
    items_sorted = items[torch.argsort(pop[items])].tolist()
    num = int(len(items_sorted) * 0.75)
    g1.extend(items_sorted[:num])
    g2.extend(items_sorted[num:])
    
    return torch.LongTensor(g1), torch.LongTensor(g2)

def InfoNCE(view1, view2, temperature):
    view1, view2 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

def InfoNCE_i(view1, view2, view3,temperature,gama):
    view1, view2,view3 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score_1 = torch.exp(ttl_score_1 / temperature).sum(dim=1)
    ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
    ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / (gama*ttl_score_2+ttl_score_1+pos_score))
    return torch.mean(cl_loss)