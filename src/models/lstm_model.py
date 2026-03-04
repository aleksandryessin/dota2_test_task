import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.data.features import build_sequence_data, CONTEXT_DIM, MAX_BAN_SEQ_LEN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + context_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, ban_seq, context):
        x = self.embed(ban_seq)
        _, (h_n, _) = self.lstm(x)
        lstm_out = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(torch.cat([lstm_out, context], dim=1))


def _train_loop(net, ban_seqs, context, targets, device, batch_size, num_epochs, lr, patience, label):
    """Manual-batching training loop — avoids DataLoader overhead on MPS."""
    import time as _time

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = len(targets)
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        t_epoch = _time.time()
        net.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = net(ban_seqs[idx], context[idx])
            loss = criterion(logits, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed_epoch = _time.time() - t_epoch
        print(f"    [{label}] epoch {epoch + 1}/{num_epochs}, loss: {avg_loss:.4f}, {elapsed_epoch:.1f}s", flush=True)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  {label} early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)
    print(f"  {label} done, best loss: {best_loss:.4f}", flush=True)


def _predict_rankings(net, ban_seqs, context, batch_size, idx2target):
    """Manual-batching inference loop."""
    net.eval()
    all_probs = []
    n = len(ban_seqs)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            logits = net(ban_seqs[start : start + batch_size], context[start : start + batch_size])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    rankings = []
    for row in all_probs:
        sorted_idx = np.argsort(row)[::-1]
        rankings.append([idx2target[j] for j in sorted_idx])
    return rankings


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, embed_dim=64, hidden_dim=128, num_epochs=20, batch_size=512,
                 learning_rate=1e-3, dropout=0.3, patience=5):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
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

        self.net = LSTMNet(
            len(hero2idx), self.embed_dim, self.hidden_dim,
            CONTEXT_DIM, len(target2idx), self.dropout,
        ).to(self.device)

        _train_loop(
            self.net, ban_seqs, context, targets,
            self.device, self.batch_size, self.num_epochs, self.lr, self.patience, "LSTM",
        )

    def predict_ranking(self, df):
        ban_seqs_np, ctx_np = build_sequence_data(df, self.hero2idx, MAX_BAN_SEQ_LEN)
        ban_seqs = torch.tensor(ban_seqs_np, dtype=torch.long, device=self.device)
        context = torch.tensor(ctx_np, dtype=torch.float32, device=self.device)
        return _predict_rankings(self.net, ban_seqs, context, self.batch_size, self.idx2target)

    def get_params(self):
        return {
            "model": "lstm",
            "device": str(self.device),
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "dropout": self.dropout,
        }
