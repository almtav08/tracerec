import torch
from torch.utils.data import DataLoader
from tracerec.algorithms.graph_based.rotate import RotatE
from tracerec.algorithms.sequential_based.sasrec import SASRecEncoder
from tracerec.data.datasets.path_dataset import PathDataset
from tracerec.data.paths.path_manager import PathManager
from tracerec.data.triples.triples_manager import TriplesManager
from tracerec.losses.supcon import SupConLoss
from tracerec.samplers.path_based_sampler import PathBasedNegativeSampler


def test_graph_embedder():
    # Create a sample triples manager with some triples
    triples = [(0, 0, 1), (1, 0, 2), (2, 0, 3)]
    triples_manager = TriplesManager(triples)

    train_x, train_y, test_x, test_y = triples_manager.split(
        train_ratio=0.8, relation_ratio=True, random_state=42, device="cpu"
    )

    sampler = PathBasedNegativeSampler(
        triples_manager,
        corruption_ratio=0.5,
        device="cpu",
        min_distance=1.0,
    )
    train_x_neg = sampler.sample(train_x, num_samples=1, random_state=42)

    transe = RotatE(
        num_entities=4,
        num_relations=1,
        embedding_dim=10,
        device="cpu",
        norm=1,
    )
    transe.compile(
        optimizer=torch.optim.Adam, criterion=torch.nn.MarginRankingLoss(margin=1.0)
    )

    # Fit the model
    transe.fit(
        train_x,
        train_x_neg,
        num_epochs=1,
        batch_size=1,
        lr=0.001,
        verbose=True,
        checkpoint_path="./transe.pth",
    )

    print(transe.history)


def test_sequential_embedder():
    paths = {
        0: [0, 1],
        1: [0, 1, 2],
        2: [0, 1, 2, 3],
        3: [1, 2],
    }
    grades = [0, 1, 1, 0]

    graph_embedder = torch.load("./transe.pth", weights_only=False)

    # Create a PathManager instance
    max_seq_length = 4
    path_manager = PathManager(paths, grades, max_seq_length, graph_embedder)

    train_x, train_y, train_masks, test_x, test_y, test_masks = path_manager.split(
        train_ratio=0.5, relation_ratio=True, random_state=42, device="cpu"
    )

    sasrec = SASRecEncoder(
        embedding_dim=10,
        max_seq_length=4,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
        device="cpu",
    )

    sasrec.compile(optimizer=torch.optim.Adam, criterion=SupConLoss())

    sasrec.fit(
        train_x,
        train_y,
        train_masks,
        num_epochs=1,
        batch_size=1,
        lr=0.001,
        verbose=True,
        checkpoint_path="./sasrec.pth",
    )

    print(sasrec.history)

if __name__ == "__main__":
    test_graph_embedder()
    # test_sequential_embedder()
