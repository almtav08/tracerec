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


def path_collate(batch):
    if len(batch[0]) == 3:
        paths, grades, masks = zip(*batch)
    else:
        paths, grades = zip(*batch)
        masks = None

    paths = torch.stack(list(paths), dim=0)
    grades = torch.tensor(grades, dtype=torch.float32)
    masks = torch.stack(list(masks), dim=0) if masks[0] is not None else None

    return paths, grades, masks


def pairwise_collate(batch):
    anchor, pos, neg = zip(*batch)
    anchor_path, anchor_mask = zip(*anchor)
    pos_path, pos_mask = zip(*pos)
    neg_path, neg_mask = zip(*neg)
    
    anchor_path = torch.stack(list(anchor_path), dim=0)
    anchor_mask = torch.stack(list(anchor_mask), dim=0)
    pos_path = torch.stack(list(pos_path), dim=0)
    pos_mask = torch.stack(list(pos_mask), dim=0)
    neg_path = torch.stack(list(neg_path), dim=0)
    neg_mask = torch.stack(list(neg_mask), dim=0)

    return anchor_path, anchor_mask, pos_path, pos_mask, neg_path, neg_mask
