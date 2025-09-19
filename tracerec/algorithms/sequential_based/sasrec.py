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
        pooling="last",
        device="cpu",
    ):
        super(SASRecEncoder, self).__init__(
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            device=device,
            pooling=pooling
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

    def forward(self, input_embs, mask=None):
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
            padding_mask = mask.bool()  # Transformer expects True for padding
        else:
            padding_mask = None

        # Pass through the transformer encoder
        out = self.encoder(x, src_key_padding_mask=padding_mask)

        # Apply pooling
        if self.pooling == "last":
            # Encuentra el índice del último token válido (no padding)
            if mask is not None:
                mask_valid = torch.logical_not(mask)
                lengths = mask_valid.sum(dim=1) - 1
                user_emb = out[torch.arange(batch_size, dtype=torch.long), lengths]
            else:
                user_emb = out[:, -1, :]
        elif self.pooling == "mean":
            if mask is not None:
                mask_valid = torch.logical_not(mask)
                denom = mask_valid.sum(dim=1, keepdim=True) - 1
                user_emb = (out * mask_valid.unsqueeze(-1)).sum(dim=1) / denom
            else:
                user_emb = out.mean(dim=1)

        return user_emb
