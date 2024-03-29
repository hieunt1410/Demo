import torch

def get_metrics(metrics, grd, pred, ks):
    tmp = {"recall": {}, "ndcg": {}}
    for k in ks:
        _, col_indices = torch.topk(pred, topk)
        row_indices = torch.zeros_like(col_indices) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indices.view(-1).to('cpu'), col_indices.view(-1).to('cpu')].view(-1, topk)
        
        tmp["recall"][k] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][k] = get_ndcg(pred, grd, is_hit, topk)
        
def get_recall(pred, grd, is_hit, topk):
    epsipon = 1e-8
    hit_cnt = is_hit.sum(dim = 1)
    num_pos = grd.sum(dim = 1)
    
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsipon)).sum().item()
    
    return [nomina, denorm]

def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.long))
        
        return hit.sum(-1)
    
    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        
        return DCG(hit, topk, device)
    
    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)
        
    num_pos = grd.sum(dim = 1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)
    
    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)
    
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()
    
    return [nomina, denorm]

if __name__ == "__main__":
    main()