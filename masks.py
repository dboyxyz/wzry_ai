import torch

def create_masks(src, trg, device):
    src_mask = (src != -1).unsqueeze(-2)
    if trg is None:
        return src_mask, None
    size = trg.size(1)
    causal = torch.ones((1, size, size), dtype=torch.bool, device=device).tril()
    return src_mask, (trg != -1).unsqueeze(-2) & causal
