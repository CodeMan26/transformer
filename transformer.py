import torch
from torch import nn
import numpy as np
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device | None
        self.dtype = torch.dtype | None
        self.W = nn.Parameter(torch.Tensor)
        nn.init.trunc_normal_(self.W)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W*x

class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=None, dtype=None))
        nn.init.trunc_normal_(self.embeddings)
        self.device=torch.device | None
        self.dtype=torch.dtype | None
    def forward(self, token_ids: torch.Tensor)->torch.Tensor:
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,device=None, dtype=None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.d_model = d_model
        self.eps = eps
        self.device = torch.device | None
        self.dtype = torch.dtype | None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        rms_norm = x / rms * self.gain
        return rms_norm
def silu(x):
    sigmoid_input = torch.sigmoid(x)
    silu = x * sigmoid_input
    return silu

class FFN(nn.Module):
    def __init(self, dim: int, d_ff):
        super().__init__()
        self.d_ff = 8 / 3 * dim
        self.w1 = Linear(dim, d_ff)
        self.w2 = Linear(d_ff, dim)
        self.w3 = Linear(dim, d_ff)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        forward pass for feed forward network
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        token_positions = token_positions.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * token_positions).flatten(3)
        return y.to(dtype)

def softmax(x: torch.Tensor, dim):
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim)

# scale dot-product attention
def scaled_dot_product_attention(q, k, v, mask=None):
    batch_size, head, seq_len, d_k = k.size()
    k_transpose = k.transpose(2, 3)
    score = (q @ k_transpose) / math.sqrt(d_k)
    # masking
    if mask is not None:
        score = score.masked_fill(mask==0, float("-inf"))
    # apply softmax
    score = softmax(score)
    # final
    v = score @ v
    return v, score

class MHA(nn.Module):
    def __init__(self, d_model: int, num_head: int, Device=None):
        super().__init__()
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_concat = Linear(d_model, d_model)
        self.n_head = num_head
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k),self.split(v)
        q, k = RotaryPositionalEmbedding(q), RotaryPositionalEmbedding(k)
        out, attention = scaled_dot_product_attention(q, k, v, mask=mask) # here we need splited q,k,v
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, x: torch.Tensor):
        """
        split tensor by number of head
        """
        batch_size, len, d_model = x.size()
        d_tensor = d_model // self.n_head
        x = x.view(batch_size, len, self.n_head, d_tensor).transpose(1,2)
        return x
    def concat(self,x: torch.Tensor):
        """
        inverse of self.split
        """
        batch_size, head, len, d_tensor = x.size()
        d_model = head * d_tensor
        x = x.transpose(1, 2).contiguous().view(batch_size, len, d_model)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = MHA(d_model, num_heads)
        self.attn_norm = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.ffn_norm = RMSNorm(d_model)
    def forward(self,x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_lenght: int, num_layers: int, d_model: int):
        super().__init__()
        self.embed = EmbeddingModel(vocab_size, ...)
        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            self.layers.append(TransformerBlock(layer_id, ...))
        self.norm = RMSNorm(d_model)
        self.output = Linear(d_model, vocab_size, bias=False)
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        mask = None
        for layer in self.layers:
            h = layer(h, start_pos, ..., mask)
        h = self.norm(h)[:, -1]
        logits = self.output(h)
        probs = torch.softmax(logits, dim=-1)
        return probs
def ce_loss(prediction: torch.FloatTensor, targets: torch.LongTensor):
    pred_max = prediction.max(dim=-1, keepdim=True).values
    prediction = prediction - pred_max
    log_prob = prediction - torch.logsumexp(prediction, dim=-1, keepdim=True)
    target_log_probs = log_prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()
