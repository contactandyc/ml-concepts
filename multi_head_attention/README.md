# Multi Head Attention

In the given code, `self.attention` is an instance of PyTorch's built-in `nn.MultiheadAttention` class. It is not implemented within the `MultiHeadAttention` class you've shown; rather, it leverages the pre-implemented multi-head attention mechanism from PyTorch's neural network library (`torch.nn`).

When `self.attention(q, k, v, attn_mask=mask)` is called in the `forward` method, it internally performs all the steps required for multi-head attention, such as the splitting of Q, K, and V into multiple heads, the scaled dot-product attention, and the final concatenation.

You can find the source code and further details of PyTorch's `nn.MultiheadAttention` in the [official PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).

In multi-head attention, the original "query" (Q), "key" (K), and "value" (V) vectors are often linearly projected into multiple different "subspaces" to create multiple sets of Q, K, and V. Each of these sets is then used to compute attention outputs individually. Finally, all these outputs are concatenated and linearly projected once again to form the final output.

The linear layers in a custom implementation of multi-head attention would project the inputs (Q, K, V) to dimensions `[batch_size, sequence_length, embed_size * num_heads]`. Each `embed_size * num_heads` dimension is then split into `num_heads` different `embed_size` vectors (one for each head).

In the built-in PyTorch `nn.MultiheadAttention`, this is done internally. So, when you use `nn.MultiheadAttention`, you don't need to worry about this manual splitting.

Here's a simplified example to demonstrate multi-head attention from scratch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.q_linear = nn.Linear(self.embed_size, self.embed_size)
        self.k_linear = nn.Linear(self.embed_size, self.embed_size)
        self.v_linear = nn.Linear(self.embed_size, self.embed_size)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        N = q.shape[0]
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]

        # Linear transformation and split into h heads
        q = self.q_linear(q).view(N, q_len, self.num_heads, self.head_dim)
        k = self.k_linear(k).view(N, k_len, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(N, v_len, self.num_heads, self.head_dim)

        # Transpose for attention computation: b x n x ql x hd
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scaled_attention = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5

        if mask is not None:
            scaled_attention += (mask * -1e5)

        attention = torch.softmax(scaled_attention, dim=-1)

        out = torch.matmul(attention, v)

        # Concatenate and pass to linear layer
        out = out.transpose(1, 2).contiguous().view(N, q_len, self.embed_size)
        out = self.fc_out(out)

        return out
```

Here, the `q_linear`, `k_linear`, and `v_linear` layers project the input to a higher dimension `embed_size`. Then we reshape these to `[N, sequence_length, num_heads, head_dim]`. Each "head" gets its own `head_dim`-sized vector, and we have `num_heads` such vectors.

The code then proceeds to calculate attention scores and produce the final output, which is concatenated across all heads and passed through a final fully-connected layer (`fc_out`).