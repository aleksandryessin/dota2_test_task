import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.models.lstm_model import get_device, _train_loop, _predict_rankings
from src.data.features import build_sequence_data, CONTEXT_DIM, MAX_BAN_SEQ_LEN


class TransformerNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, context_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(MAX_BAN_SEQ_LEN, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + context_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, ban_seq, context):
        B, L = ban_seq.shape
        positions = torch.arange(L, device=ban_seq.device).unsqueeze(0).expand(B, -1)
        x = self.embed(ban_seq) + self.pos_embed(positions)

        padding_mask = ban_seq == 0
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)

        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.head(torch.cat([pooled, context], dim=1))


class TransformerModel(BaseModel):
    name = "transformer"

    def __init__(self, embed_dim=64, nhead=4, num_layers=2, num_epochs=20,
                 batch_size=512, learning_rate=1e-3, dropout=0.3, patience=5):
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.dropout = dropout
        self.patience = patience
        self.net = None
        self.hero2idx = None
        self.target2idx = None
        self.idx2target = None
        self.device = get_device()

    def fit(self, train_df, hero2idx, idx2hero, target2idx, idx2target):
        self.hero2idx = hero2idx
        self.target2idx = target2idx
        self.idx2target = idx2target

        print(f"  device: {self.device}", flush=True)

        ban_seqs_np, ctx_np = build_sequence_data(train_df, hero2idx, MAX_BAN_SEQ_LEN)
        tgt_np = train_df["first_pick_hero"].map(target2idx).values

        ban_seqs = torch.tensor(ban_seqs_np, dtype=torch.long, device=self.device)
        context = torch.tensor(ctx_np, dtype=torch.float32, device=self.device)
        targets = torch.tensor(tgt_np, dtype=torch.long, device=self.device)

        self.net = TransformerNet(
            len(hero2idx), self.embed_dim, self.nhead, self.num_layers,
            CONTEXT_DIM, len(target2idx), self.dropout,
        ).to(self.device)

        _train_loop(
            self.net, ban_seqs, context, targets,
            self.device, self.batch_size, self.num_epochs, self.lr, self.patience, "Transformer",
        )

    def predict_ranking(self, df):
        ban_seqs_np, ctx_np = build_sequence_data(df, self.hero2idx, MAX_BAN_SEQ_LEN)
        ban_seqs = torch.tensor(ban_seqs_np, dtype=torch.long, device=self.device)
        context = torch.tensor(ctx_np, dtype=torch.float32, device=self.device)
        return _predict_rankings(self.net, ban_seqs, context, self.batch_size, self.idx2target)

    def get_params(self):
        return {
            "model": "transformer",
            "device": str(self.device),
            "embed_dim": self.embed_dim,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "dropout": self.dropout,
        }
