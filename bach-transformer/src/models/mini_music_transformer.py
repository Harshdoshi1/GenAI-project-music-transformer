from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniMusicTransformer(nn.Module):
    """A tiny autoregressive Transformer LM with learned absolute positions.
    Designed to be lightweight and easy to train on JSB chorales tokenized as note ids.
    """

    def __init__(
        self,
        vocab_size: int = 133,  # 5 special tokens + 128 pitches
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """Args:
        - idx: (B, T) Long
        - attn_mask: (T, T) with -inf for masked (causal), added to attention weights
        Returns logits: (B, T, vocab)
        """
        B, T = idx.shape
        assert T <= self.max_len, "Sequence length exceeds model max_len"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x, mask=attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
        self.eval()
        for _ in range(max_new_tokens):
            if idx.size(1) > self.max_len:
                idx = idx[:, -self.max_len :]
            T = idx.size(1)
            mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float("-inf"))
            logits = self(idx, attn_mask=mask)[:, -1, :] / max(1e-6, temperature)

            if top_k and top_k > 0:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
