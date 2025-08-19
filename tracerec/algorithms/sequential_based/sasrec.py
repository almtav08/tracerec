import torch
import torch.nn as nn

from .sequential_embedder import SequentialEmbedder


class SASRecEncoder(SequentialEmbedder):
    def __init__(
        self,
        embedding_dim,
        max_seq_length,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
        device="cpu",
    ):
        super(SASRecEncoder, self).__init__(
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            device=device,
        )

        self.num_layers = num_layers
        self.num_heads = num_heads

        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_device(device)

    def to_device(self, device="cpu"):
        self.position_embedding = self.position_embedding.to(device)
        self.encoder = self.encoder.to(device)
        return super().to_device(device)

    def forward(self, input_embs, mask=None, pooling="mean"):
        """
        Forward pass through the SASRec encoder.
        Args:
            input_embs: Tensor of shape (batch_size, seq_length, embedding_dim)
            mask: Optional mask tensor for padding
            pooling: Pooling method to apply to the output
        Returns:
            Tensor of shape (batch_size, embedding_dim) after pooling
        """
        batch_size, seq_length, _ = input_embs.shape

        # Add positional embeddings
        positions = torch.arange(seq_length, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        position_embs = self.position_embedding(positions)

        # Combine input embeddings with positional embeddings
        x = input_embs + position_embs

        # Create mask for the Transformer if padding is present
        if mask is not None:
            attention_mask = ~mask.bool()  # Transformer expects True for padding
        else:
            attention_mask = None

        # Pass through the transformer encoder
        out = self.encoder(x, src_key_padding_mask=attention_mask)

        # Apply pooling
        if pooling == "last":
            # Encuentra el índice del último token válido (no padding)
            if mask is not None:
                lengths = mask.sum(dim=1) - 1
                user_emb = out[torch.arange(batch_size, dtype=torch.long), lengths]
            else:
                user_emb = out[:, -1, :]
        elif pooling == "mean":
            if mask is not None:
                user_emb = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                    dim=1, keepdim=True
                )
            else:
                user_emb = out.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

        return user_emb
