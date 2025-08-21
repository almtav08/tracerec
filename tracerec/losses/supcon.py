import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import Loss


class SupConLoss(Loss):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.pairwise = False

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Normalizar embeddings
        features = F.normalize(features, dim=1)

        # Similaridades
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Máscara de clases
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Evitar dividir por 0
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # Pérdida
        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return loss.mean()
