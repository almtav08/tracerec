import torch
from tracerec.algorithms.knowledge_based.transe import TransE
from tracerec.data.triples.triples_manager import TriplesManager
from tracerec.samplers.path_based_sampler import PathBasedNegativeSampler


def test_triples_dataset():
    # Create a sample triples manager with some triples
    triples = [
        (1, 0, 2),
        (2, 0, 3),
        (3, 0, 4)
    ]
    triples_manager = TriplesManager(triples)

    train_x, train_y, test_x, test_y = triples_manager.split(train_ratio=0.8, relation_ratio=True, random_state=42, device='cpu')

    # Negative sampling
    all_triples = triples_manager.get_triples()
    all_entities = triples_manager.get_entities()
    entity_paths = triples_manager.get_entity_paths()

    sampler = PathBasedNegativeSampler(all_triples, all_entities, corruption_ration=0.5, device='cpu', entity_paths=entity_paths, min_distance=1.0)
    train_x_neg = sampler.sample(train_x, num_samples=1, random_state=42)

    # Create and compile the TransE model
    transe = TransE(num_entities=4, num_relations=1, embedding_dim=10, device='cpu', norm=1)
    transe.compile(optimizer=torch.optim.Adam, criterion=torch.nn.MarginRankingLoss(margin=1.0))

    # Fit the model
    transe.fit(train_x, train_x_neg, train_y, num_epochs=1, batch_size=1, lr=0.001, verbose=True)

    print(transe.history)

if __name__ == "__main__":
    test_triples_dataset()
