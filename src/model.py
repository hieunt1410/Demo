import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_bpr_loss(pred):
    negs =  pred[:, 1].unsqueeze(1)
    pos = pred[:, 0].unsqueeze(1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos - negs)))
    
    return loss

def laplace_transform(graph):
    epsilon = 1e-8
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + epsilon))
    colsum_sqty = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + epsilon))
    graph = rowsum_sqrt @ graph @ colsum_sqty
    
    return graph

def to_tensor(graph):
    """Convert to sparse tensor"""
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    
    return graph

def np_edge_dropout(values, dropout):
    mask = np.random.choice([0, 1], size = len(values), p=[dropout, 1 - dropout])
    values = mask * values
    
    return values

class Demo(nn.Module):
    def __init__(self, conf, raw_graph, bundles_freq):
        super(Demo, self).__init__()
        self.conf = conf
        self.device = conf['device']
        self.embedding_size = conf['embedding_size']
        self.l2_norm = conf['lambda2']
        self.num_users = conf['num_users']
        self.num_bundles = conf['num_bundles']
        self.num_items = conf['num_items']
        self.num_layers = conf['num_layers']
        
        self.bundle_freq = torch.FloatTensor(bundles_freq).to(self.device)
        
        self.init_embed()
        
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        self.get_aff_graph()
        self.get_hist_graph()
        self.get_agg_graph()
        
        self.get_aff_graph_ori()
        self.get_hist_graph_ori()
        self.get_agg_graph_ori()
        
    def init_embed(self):
        self.users_feat = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feat)
        self.bundles_feat = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feat)
        self.items_feat = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feat)
        self.IL_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.IL_layer.weight)
        self.BL_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.BL_layer.weight)
        
    def get_aff_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf['aff_ed_ratio']
        
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio:
            graph = item_level_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
            
        self.aff_view_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
        
    def get_aff_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.aff_view_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)
    
    def get_hist_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf['hist_ed_ratio']
        
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        
        if modification_ratio:
            graph = bundle_level_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        self.hist_view_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)
        
    def get_hist_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        
        self.hist_view_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)
        
    def get_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device
        modification_ratio = self.conf['agg_ed_ratio']
        
        graph = bi_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        bundle_sz = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_sz.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)
        
    def get_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device
        
        bundle_sz = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_sz.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)
        
    def one_propagate(self, graph, Afeat, Bfeat):
        feats = torch.cat((Afeat, Bfeat), dim=0)
        all_feats = [feats]
        
        for i in range(self.num_layers):
            feats = torch.spmm(graph, feats)
            feats = feats / (i+2)
            all_feats.append(F.normalize(feats, p=2, dim=1))
            
        all_feats = torch.stack(all_feats, dim=1)
        all_feats = torch.sum(all_feats, dim=1).squeeze(1)
        
        Afeat, Bfeat = torch.split(all_feats, (Afeat.shape[0], Bfeat.shape[0]))
        
        return Afeat, Bfeat
    
    def get_aff_bundle_rep(self, aff_items_feat, test):
        if test:
            aff_bundles_feat = self.bundle_agg_graph_ori @ aff_items_feat
        else:
            aff_bundles_feat = self.bundle_agg_graph @ aff_items_feat

        return aff_bundles_feat
    
    def propagate(self, test=False):
        if test:
            aff_users_feat, aff_items_feat = self.one_propagate(self.aff_view_graph_ori, self.users_feat, self.items_feat)
            
        else:
            aff_users_feat, aff_items_feat = self.one_propagate(self.aff_view_graph, self.users_feat, self.items_feat)
            
        if test:
            hist_users_feat, hist_bundles_feat = self.one_propagate(self.hist_view_graph_ori, self.users_feat, self.bundles_feat)
        else:
            hist_users_feat, hist_bundles_feat = self.one_propagate(self.hist_view_graph, self.users_feat, self.bundles_feat)
            
        aff_bundles_feat = self.get_aff_bundle_rep(aff_items_feat, test)
        
        users_feat = [aff_users_feat, hist_users_feat]
        bundles_feat = [aff_bundles_feat, hist_bundles_feat]
        
        return users_feat, bundles_feat
    
    def cal_a_loss(self, x, y):
        x, y = F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1)       
        return (x - y).norm(p=2, dim=1).pow(2).mean()
    
    def cal_u_loss(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
    
    def cal_loss(self, users_feat, bundles_feat, bundles_gamma):
        aff_users_feat, hist_users_feat = users_feat
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        aff_bundles_feat_ = aff_bundles_feat * (1 - bundles_gamma).unsqueeze(2)
        hist_bundles_feat_ = hist_bundles_feat * bundles_gamma.unsqueeze(2)
        
        pred = torch.sum(aff_bundles_feat * aff_bundles_feat_, 2) + torch.sum(hist_bundles_feat * hist_bundles_feat_, 2)
        bpr_loss = cal_bpr_loss(pred)
        
        aff_bundles_feat = aff_bundles_feat[:, 0, :]
        hist_bundles_feat = hist_bundles_feat[:, 0, :]
        bundle_align = self.cal_a_loss(aff_bundles_feat, hist_bundles_feat)
        bundle_uniform = self.cal_u_loss(aff_bundles_feat) + self.cal_u_loss(hist_bundles_feat) / 2
        bundle_c_loss = bundle_align + bundle_uniform
        
        aff_users_feat = aff_users_feat[:, 0, :]
        hist_users_feat = hist_users_feat[:, 0, :]
        user_align = self.cal_a_loss(aff_users_feat, hist_users_feat)
        user_uniform = self.cal_u_loss(aff_users_feat) + self.cal_u_loss(hist_users_feat) / 2
        user_c_loss = user_align + user_uniform
        
        c_loss = (bundle_c_loss + user_c_loss) / 2
        
        return bpr_loss, c_loss
    
    def forward(self, batch, ED_dropout, psi=1.):
        if ED_dropout:
            self.get_aff_graph()
            self.get_hist_graph()
            self.get_agg_graph()
        
        users, bundles = batch
        users_feat, bundles_feat = self.propagate()
        
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feat]
        bundles_embedding = [i[bundles] for i in bundles_feat]
        bundles_gamma = torch.tanh(self.bundle_freq / psi)
        bundles_gamma = bundles_gamma[bundles.flatten()].reshape(bundles.shape)
                                                                 
        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding, bundles_gamma)
        
        return bpr_loss, c_loss
        
    def evaluate(self, propagate_result, users, psi=1):
        users_feat, bundles_feat = propagate_result
        aff_users_feat, hist_users_feat = [i[users] for i in users_feat]
        aff_bundles_feat, hist_bundles_feat = bundles_feat
        bundle_gamma = torch.tanh(self.bundle_freq / psi)
        aff_bundles_feat_ =  aff_bundles_feat * (1 - bundle_gamma.unsqueeze(1))
        hist_bundles_feat_ = hist_bundles_feat * bundle_gamma.unsqueeze(1)
        scores = aff_bundles_feat @ aff_bundles_feat_.t() + hist_bundles_feat @ hist_bundles_feat_.t()
        
        return scores