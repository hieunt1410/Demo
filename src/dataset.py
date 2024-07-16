import os

import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import numpy as np

class TrainDataset(Dataset):
    """
    Class of training dataset
    """
    def __init__(self, conf, ub_pairs, ub_graph, num_bundles, neg_sample=1):
        self.conf = conf
        self.ub_pairs = ub_pairs
        self.ub_graph = ub_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample
    
    def __len__(self):
        return len(self.ub_pairs)
    
    def __getitem__(self, idx):
        u, pos_b = self.ub_pairs[idx]
        all_b = [pos_b]
        
        while len(all_b) < self.neg_sample + 1:
            neg_b = np.random.randint(self.num_bundles)
            if self.ub_graph[u, neg_b] == 0 and not neg_b in all_b:
                all_b.append(neg_b)
        
        return torch.LongTensor([u]), torch.LongTensor(all_b)
    
class TestDataset(Dataset):
    """
    Class of test dataloader
    """
    def __init__(self, ub_pairs, ub_graph, ub_graph_train, num_users, num_bundles):
        self.ub_pairs = ub_pairs
        self.ub_graph = ub_graph
        self.train_mask_ub = ub_graph_train
        
        self.num_users = num_users
        self.num_bundles = num_bundles
        
        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)
        
    def __len__(self):
        return self.ub_graph.shape[0]
        
    def __getitem__(self, idx):
        ub_grd = torch.from_numpy(self.ub_graph[idx].toarray()).squeeze()
        ub_mask = torch.from_numpy(self.train_mask_ub[idx].toarray()).squeeze()
        
        return idx, ub_grd, ub_mask
    
class Datasets():
    """
    Model datasets
    """
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        bsz_train = conf['batch_size_train']
        bsz_test = conf['batch_size_test']
        
        self.num_users, self.num_bundles, self.num_items = self.get_data_size()
        bi_graph = self.get_bi()
        ui_pairs, ui_graph = self.get_ui()
        
        ub_pairs_train, ub_graph_train = self.get_ub('train')
        ub_pairs_val, ub_graph_val = self.get_ub('tune')
        ub_pairs_test, ub_graph_test = self.get_ub('test')
        
        self.bundle_train_data = TrainDataset(conf, ub_pairs_train, ub_graph_train, self.num_bundles)
        self.bundle_val_data = TestDataset(ub_pairs_val, ub_graph_val, ub_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = TestDataset(ub_pairs_test, ub_graph_test, ub_graph_train, self.num_users, self.num_bundles)
        
        self.graphs = [ub_graph_train, ui_graph, bi_graph]
        
        self.train_loader = DataLoader(self.bundle_train_data, batch_size=bsz_train, shuffle=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=bsz_test, shuffle=False)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=bsz_test, shuffle=False)
        
        self.bundles_freq = np.asarray(ub_graph_train.sum(axis=0)).squeeze()
        
    def get_data_size(self):
        name = self.name
        with open(os.path.join(self.path, name, f'{name}_data_size.txt'), 'r') as f:
            return [int(s) for s in f.readline().strip().split('\t')]
    
    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            bi_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        
        indices = np.array(bi_pairs, dtype=np.int32)
        values = np.ones(len(bi_pairs))
        bi_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()
        
        return bi_graph
    
    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            ui_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
            
        indices = np.array(ui_pairs, dtype=np.int32)
        values = np.ones(len(ui_pairs))
        ui_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=(self.num_users, self.num_items)).tocsr()
        
        return ui_pairs, ui_graph[:10000]
    
    def get_ub(self, mode):
        with open(os.path.join(self.path, self.name, f'user_bundle_{mode}.txt'), 'r') as f:
            ub_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        
        indices = np.array(ub_pairs, dtype=np.int32)
        values = np.ones(len(ub_pairs))
        ub_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()
        
        return ub_pairs, ub_graph[:10000]