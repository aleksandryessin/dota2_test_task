"""Candidate-scoring models with prior integration.

V1 (CandidateScorerNet): mean-pool ban encoder + dot-product scoring.
V2 (CrossAttentionScorerNet): hero candidates attend to ban tokens via cross-attention.
"""

import torch
import torch.nn as nn


class CandidateScorerNet(nn.Module):
    def __init__(self, vocab_size, num_heroes, embed_dim, hidden_dim,
                 context_dim, num_layers, nhead, dropout):
        super().__init__()
        self.num_heroes = num_heroes

        self.ban_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.ban_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ban_norm = nn.LayerNorm(embed_dim)

        self.ctx_proj = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.match_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.hero_embed = nn.Embedding(num_heroes, embed_dim)

        self.prior_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.5, 0.3]))
        self.bias = nn.Parameter(torch.zeros(num_heroes))

    def forward(self, ban_seq, context, cap_prior, team_prior, meta_prior,
                series_prior=None):
        pad_mask = ban_seq == 0
        x = self.ban_embed(ban_seq)
        x = self.ban_encoder(x, src_key_padding_mask=pad_mask)
        x = self.ban_norm(x)

        mask = (~pad_mask).unsqueeze(-1).float()
        ban_repr = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        ctx_repr = self.ctx_proj(context)
        match_repr = self.match_head(torch.cat([ban_repr, ctx_repr], dim=1))

        hero_embs = self.hero_embed.weight
        dot_scores = match_repr @ hero_embs.T

        log_cap = torch.log(cap_prior + 1e-8)
        log_team = torch.log(team_prior + 1e-8)
        log_meta = torch.log(meta_prior + 1e-8)

        w = nn.functional.softplus(self.prior_weights)
        prior_scores = w[0] * log_cap + w[1] * log_team + w[2] * log_meta

        if series_prior is not None:
            prior_scores = prior_scores + w[3] * series_prior

        return dot_scores + prior_scores + self.bias


class CrossAttentionScorerNet(nn.Module):
    """Hero candidates attend to ban tokens via cross-attention.

    Memory = [ban_token_1, ..., ban_token_L, ctx_token]  (L+1 tokens)
    Queries = hero candidate embeddings  (num_heroes tokens)
    Cross-attention → per-hero refined representations → scalar scores.
    """

    def __init__(self, vocab_size, num_heroes, embed_dim, hidden_dim,
                 context_dim, num_layers, nhead, dropout, n_cross_layers=1):
        super().__init__()
        self.num_heroes = num_heroes
        self.embed_dim = embed_dim

        self.ban_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.ban_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ban_norm = nn.LayerNorm(embed_dim)

        self.ctx_proj = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.hero_embed = nn.Embedding(num_heroes, embed_dim)

        cross_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.cross_decoder = nn.TransformerDecoder(cross_layer, num_layers=n_cross_layers)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.prior_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.5, 0.3]))
        self.bias = nn.Parameter(torch.zeros(num_heroes))

    def forward(self, ban_seq, context, cap_prior, team_prior, meta_prior,
                series_prior=None):
        B = ban_seq.size(0)

        pad_mask = ban_seq == 0
        x = self.ban_embed(ban_seq)
        x = self.ban_encoder(x, src_key_padding_mask=pad_mask)
        x = self.ban_norm(x)

        ctx_token = self.ctx_proj(context).unsqueeze(1)  # [B, 1, D]
        memory = torch.cat([x, ctx_token], dim=1)  # [B, L+1, D]

        mem_pad = torch.cat([
            pad_mask,
            torch.zeros(B, 1, device=ban_seq.device, dtype=torch.bool),
        ], dim=1)  # [B, L+1]

        hero_queries = self.hero_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, H, D]

        refined = self.cross_decoder(hero_queries, memory, memory_key_padding_mask=mem_pad)
        refined = self.cross_norm(refined)  # [B, H, D]

        scores = self.score_head(refined).squeeze(-1)  # [B, H]

        log_cap = torch.log(cap_prior + 1e-8)
        log_team = torch.log(team_prior + 1e-8)
        log_meta = torch.log(meta_prior + 1e-8)

        w = nn.functional.softplus(self.prior_weights)
        prior_scores = w[0] * log_cap + w[1] * log_team + w[2] * log_meta

        if series_prior is not None:
            prior_scores = prior_scores + w[3] * series_prior

        return scores + prior_scores + self.bias
