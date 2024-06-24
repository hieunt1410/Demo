import pandas as pd
import os
import sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

def gen_bun_attention_graph(dataset, path):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, bi_graph, ui_graph, _ = dataset.graphs
    
    graph = bi_graph.tocoo()
    be = []
    for b in range(bi_graph.shape[0]):
        idx = bi_graph[b].nonzero()[1]
        w = F.softmax(torch.Tensor(ui_graph.T[idx].sum(axis=1).tolist()), 0).to(device)
        be += w.reshape(1, -1).tolist()[0]

    bi_graph = sp.coo_matrix((be, (graph.row, graph.col)), shape=graph.shape).tocsr()

    # if modification_ratio:
    #     graph = bi_graph.tocoo()
    #     values = np_edge_dropout(graph.data, modification_ratio)
    #     bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
    
    # return to_tensor(bi_graph).to(device)
    with open(path, 'wb') as f:
        pickle.dump(bi_graph, f)