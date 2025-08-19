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
    masks = torch.stack(list(masks), dim=0) if masks is not None else None

    return paths, grades, masks
