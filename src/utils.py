import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def mix_graph(raw_graph, num_users, num_items, num_bundles, threshold=10):
    ub_graph, ui_graph, bi_graph = raw_graph
    
    ii_graph = np.zeros((num_items, num_items), dtype=np.int32)
    
    uu_graph = ub_graph @ ub_graph.T
    for i in range(ub_graph.shape[0]):
        for r in range(uu_graph.indptr[i], uu_graph.indptr[i+1]):
            uu_graph.data[r] = 1 if uu_graph.data[r] > threshold else 0
    
    
    bb_graph = ub_graph.T @ ub_graph
    for i in range(ub_graph.shape[1]):
        for r in range(bb_graph.indptr[i], bb_graph.indptr[i+1]):
            bb_graph.data[r] = 1 if bb_graph.data[r] > threshold else 0
            
    uu_graph = uu_graph + np.eye(uu_graph.shape[0])
    bb_graph = bb_graph + np.eye(bb_graph.shape[0])
    
    H1 = sp.hstack([uu_graph, ui_graph, ub_graph])
    H2 = sp.hstack([ui_graph.T, ii_graph, bi_graph.T])
    H3 = sp.hstack([ub_graph.T, bi_graph, bb_graph])
    H = sp.vstack([H1, H2, H3])
    print('Finish mix hypergraph')
    
    return H

def normalize_Hyper(H):
    D_v = sp.diags(1 / (np.sqrt(H.sum(axis=1).A.ravel()) + 1e-8))
    D_e = sp.diags(1 / (np.sqrt(H.sum(axis=0).A.ravel()) + 1e-8))
    H_nomalized = D_v @ H @ D_e @ H.T @ D_v
    return H_nomalized

def split_hypergraph(H, device, split_num=16):
    H_list = []
    length = H.shape[0] // split_num
    
    for i in range(split_num):
        if i == split_num - 1:
            H_list.append(H[length*i:, :])
        else:
            H_list.append(H[length*i:length*(i+1)])
    
    H_split = [torch.tensor(h_i).to_sparse().to(device) for H_i in H_list]
    
    return H_split