import random

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
import pprint

from model import *
from dataset import Datasets
from metrics import get_metrics

# Path: src/main.py
def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='Youshu', help='dataset name')
    parser.add_argument('--model', '-m', type=str, default='BPR', help='model name')
    
    args = parser.parse_args()
    
    return args

def main():
    conf = yaml.safe_load(open('../configs/config.yaml', 'r'))
    print('Config file loaded')
    
    params = get_cmd().__dict__
    dataset_name = params['data']
    conf = conf[dataset_name]
    
    conf['dataset'] = dataset_name
    conf['model'] = params['model']
    dataset = Datasets(conf)
    
    conf['num_users'] = dataset.num_users
    conf['num_bundles'] = dataset.num_bundles
    conf['num_items'] = dataset.num_items
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf['device'] = device
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    pp.pprint(conf)
    
    torch.manual_seed(2024)
    np.random.seed(2024)
    
    model = Demo(conf, dataset.graphs, dataset.bundles_freq).to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf['lr'], weight_decay=conf['lambda2'])
    topk_ = conf['topk_valid']
    best_vld_rec, best_vld_ndcg, best_content = 0, 0, ''
    
    for epoch in range(1, conf['epochs']+1):
        model.train(True)
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        cur_instance_num, loss_avg, bpr_loss_avg, c_loss_avg = 0., 0., 0., 0.
        mult = epoch / conf['epochs']
        psi = conf['max_temp'] ** mult
        
        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            
            bpr_loss, c_loss = model(batch, ED_dropout=True, psi=psi)
            loss = bpr_loss + c_loss * conf['lambda1']
            loss.backward()
            optimizer.step()
            
            loss_scalar = loss.detach()
            bpr_loss_scalar = bpr_loss.detach()
            c_loss_scalar = c_loss.detach()
            
            loss_avg = moving_avg(loss_avg, cur_instance_num, loss_scalar, batch[0].shape[0])
            bpr_loss_avg = moving_avg(bpr_loss_avg, cur_instance_num, bpr_loss_scalar, batch[0].shape[0])
            c_loss_avg = moving_avg(c_loss_avg, cur_instance_num, c_loss_scalar, batch[0].shape[0])
            cur_instance_num += batch[0].shape[0]
            pbar.set_description(f'Epoch {epoch} | BPR Loss: {bpr_loss_avg:.4f} | Content Loss: {c_loss_avg:.4f} | Total Loss: {loss_avg:.4f}')
            
        if epoch % conf['test_interval'] == 0:
            metrics = {}
            metrics['val'] = test(model, dataset.val_loader, conf, psi)
            metrics['test'] = test(model, dataset.test_loader, conf, psi)
            content = form_content(epoch, metrics['val'], metrics['test'], conf['topk'])
            print(content)
            if metrics['val']['recall'][topk_] > best_vld_rec and metrics['val']['ndcg'][topk_] > best_vld_ndcg:
                best_vld_rec = metrics['val']['recall'][topk_]
                best_vld_ndcg = metrics['val']['ndcg'][topk_]
                best_content = content
                if not os.path.exists('../models'):
                    os.makedirs('../models')
                torch.save(model.state_dict(), f'../models/{conf["model"]}_{conf["dataset"]}.pth')
    
    print('-'*20, 'Best Result', '-'*20)
    print(best_content)
    
def moving_avg(avg, cur_num, add_value_avg, add_num):
    return (avg * cur_num + add_value_avg * add_num) / (cur_num + add_num)

def form_content(epoch, val_results, test_results, ks):
    content = f'     Epoch|  Rec@{ks[0]} |  Rec@{ks[1]} |  Rec@{ks[2]} |  Rec@{ks[3]} |' \
             f' nDCG@{ks[0]} | nDCG@{ks[1]} | nDCG@{ks[2]} | nDCG@{ks[3]} |\n'
    val_content = f'{epoch:10d}|'
    val_results_recall = val_results['recall']
    for k in ks:
        val_content += f'  {val_results_recall[k]:.4f} |'
    val_results_ndcg = val_results['ndcg']
    for k in ks:
        val_content += f'  {val_results_ndcg[k]:.4f} |'
    content += val_content + '\n'
    test_content = f'{epoch:10d}|'
    test_results_recall = test_results['recall']
    for k in ks:
        test_content += f'  {test_results_recall[k]:.4f} |'
    test_results_ndcg = test_results['ndcg']
    for k in ks:
        test_content += f'  {test_results_ndcg[k]:.4f} |'
    content += test_content
    
    return content
    
def test(model, dataloader, conf, tau=1.0):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device), tau)
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics

if __name__ == '__main__':
    main()