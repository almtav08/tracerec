import torch


def pos_neg_triple_collate(batch):
    """
    Custom collate function to handle batches of triples with their negatives.
    """
    _, neg_triples = zip(*batch)

    output = torch.zeros(
        len(batch), 3 * (neg_triples[0].size()[0] + 1), dtype=torch.long
    )  # Assuming triples are in (subject, relation, object) format

    for i, (pos, neg) in enumerate(batch):
        output[i, :3] = pos.detach().clone()
        output[i, 3 : 3 + 3 * len(neg)] = torch.tensor(
            neg.view(-1).tolist(), dtype=torch.long
        )

    return output
