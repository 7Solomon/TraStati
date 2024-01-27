import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nhead=8, num_encoder_layers=6, num_positions=20):
        super(DETR, self).__init__()

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead),
            num_layers=num_encoder_layers
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, num_positions)

        # Output layers
        self.linear_bbox = nn.Linear(hidden_dim, num_positions * 4)  # [x, y, type, degree] for each position
        self.linear_class = nn.Linear(hidden_dim, num_positions * num_classes)  # Class prediction for each position

    def forward(self, x):
        # Simple CNN backbone
        x = self.backbone(x)

        # Flatten spatial dimensions
        x = x.flatten(2).permute(2, 0, 1)

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Predictions
        bbox_preds = self.linear_bbox(x).view(x.size(0), -1, 4)  # Reshape to [batch_size, num_positions, 4]
        class_preds = self.linear_class(x).view(x.size(0), -1, num_classes)  # Reshape to [batch_size, num_positions, num_classes]

        return bbox_preds, class_preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_positions):
        super(PositionalEncoding, self).__init__()

        max_len = num_positions
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term[:max_len])
        pe[:, 0, 1::2] = torch.cos(position * div_term[:max_len])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
