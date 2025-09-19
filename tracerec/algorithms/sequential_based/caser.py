import torch
import torch.nn as nn

from .sequential_embedder import SequentialEmbedder


class CaserEncoder(SequentialEmbedder):
    def __init__(
        self,
        embedding_dim,
        max_seq_length,
        num_vert_filters=4,
        num_hori_filters=16,
        hori_filter_sizes=(2, 3, 4),
        pooling="last",
        dropout=0.2,
        device="cpu",
    ):
        super(CaserEncoder, self).__init__(
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pooling=pooling,
            device=device,
        )

        self.horizontal_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_hori_filters,
                    kernel_size=(h, embedding_dim),
                )
                for h in hori_filter_sizes
            ]
        )
        hori_out_dim = num_hori_filters * len(hori_filter_sizes)

        self.vertical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_vert_filters,
            kernel_size=(max_seq_length, 1),
        )
        vert_out_dim = num_vert_filters * embedding_dim

        self.fc = nn.Linear(hori_out_dim + vert_out_dim, embedding_dim)

        self.to_device(device)

    def to_device(self, device="cpu"):
        self.horizontal_convs = self.horizontal_convs.to(device)
        self.vertical_conv = self.vertical_conv.to(device)
        self.fc = self.fc.to(device)
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
        batch_size, seq_length, emb_dim = input_embs.size()

        # --- Opcional: aplicar máscara de padding ---
        if mask is not None:
            mask_valid = torch.logical_not(mask)
            input_embs = input_embs * mask_valid.unsqueeze(-1)

        # Añadimos una dimensión de canal para convoluciones 2D
        # shape: (batch_size, 1, seq_length, emb_dim)
        x = input_embs.unsqueeze(1)

        # --- Convoluciones horizontales ---
        conv_h = []
        for conv in self.horizontal_convs:
            c_out = torch.relu(conv(x)).squeeze(
                3
            )  # -> (batch_size, num_filters, seq_length - k + 1)
            p_out = torch.max_pool1d(c_out, c_out.size(2)).squeeze(
                2
            )  # -> (batch_size, num_filters)
            conv_h.append(p_out)
        z_h = torch.cat(conv_h, dim=1)  # concatenar filtros horizontales

        # --- Convoluciones verticales ---
        # Conv vertical = kernel de tamaño (seq_length, 1), colapsa en una sola fila
        z_v = torch.relu(self.vertical_conv(x)).view(
            batch_size, -1
        )  # -> (batch_size, emb_dim * num_vertical_filters)

        # Concatenar representaciones horizontales y verticales
        z = torch.cat([z_h, z_v], dim=1)

        # Pasar por fully connected para obtener embedding de usuario
        user_emb = self.fc(z)

        return user_emb
