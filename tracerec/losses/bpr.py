import torch
import torch.nn as nn

from .loss import Loss


class BPRLoss(Loss):
    def __init__(self, l2_reg=1e-4):
        super().__init__()
        self.l2_reg = l2_reg
        self.pairwise = True
        
    def forward(self, anchor_embs, pos_user_embs, neg_user_embs):
        pos_scores = (anchor_embs * pos_user_embs).sum(dim=1)
        neg_scores = (anchor_embs * neg_user_embs).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        if self.l2_reg > 0:
            l2 = anchor_embs.norm(2).pow(2) / anchor_embs.size(0)
            loss += self.l2_reg * l2
        return loss
