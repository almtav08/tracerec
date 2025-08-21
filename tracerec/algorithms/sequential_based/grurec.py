import torch
import torch.nn as nn

from .sequential_embedder import SequentialEmbedder


class GRU4Rec(SequentialEmbedder):
    def __init__(
        self,
        embedding_dim,
        max_seq_length=-1,
        items_dim=100,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        pooling="last",
        device="cpu",
    ):
        super(GRU4Rec, self).__init__(
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            device=device,
            pooling=pooling,
        )

        self.gru = nn.GRU(
            items_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, embedding_dim)

        self.to_device(device)

    def to_device(self, device="cpu"):
        self.gru = self.gru.to(device)
        self.fc = self.fc.to(device)
        return super().to_device(device)

    def forward(self, seq_emb, mask=None):
        """
        seq_emb: [batch_size, seq_len, embed_dim]
                 embeddings precomputados de los ítems en la secuencia
        """
        output, _ = self.gru(seq_emb)

        if self.pooling == "last":
            user_emb = output[:, -1, :]

        elif self.pooling == "mean":
            user_emb = output.mean(dim=1)

        user_emb = self.fc(user_emb)  # [batch_size, embed_dim]

        return user_emb
