# Flexible Phrase and Frequencies

Frequency and phase are essential components of a sinusoidal function.

- **Frequency**: It represents how many times a wave oscillates (goes up and down) within a given period. Higher frequency means more oscillations and therefore a quicker repetition of values.

- **Phase**: It indicates the initial angle (or initial point) of the oscillation in its wave cycle. The phase shifts the wave horizontally, which can be especially useful for distinguishing signals that have the same frequency but different starting points.

The flexibility of frequencies and phases in the sine and cosine functions enriches positional encodings, offering an additional layer of nuance. Varying frequencies and phases allow the model to capture both local and global relationships in the data sequence. This adaptability aids the model in generalizing well to various types of data.

In mathematical terms, the varying frequencies are dictated by the term \( \frac{1}{10000^{2i/d}} \) in the sinusoidal encoding equation. This introduces different frequencies and implicitly different phases, enabling the model to capture different types of positional dependencies.

Here's a PyTorch code snippet to illustrate:

```python
import torch
import math
import matplotlib.pyplot as plt

# Initialize sequence length and dimensions
seq_len = 100
dim = 16

# Function to generate sinusoidal positional encodings
def sinusoidal_encoding(seq_len, dim):
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
    pos_enc = pos * div_term
    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])
    return pos_enc

# Generate sinusoidal encoding
encoding = sinusoidal_encoding(seq_len, dim)

# Plot positional encodings for selected dimensions to visualize frequencies and phases
plt.figure(figsize=(16, 8))
for i in range(0, 8, 2):  # Sample dimensions
    plt.subplot(4, 1, i // 2 + 1)
    plt.title(f"Dimension {i} and {i + 1}")
    plt.plot(encoding[:, i], label=f"Sine Dimension {i} (Freq: {1 / math.exp(i * -(math.log(10000.0) / dim))})")
    plt.plot(encoding[:, i + 1], label=f"Cosine Dimension {i + 1} (Freq: {1 / math.exp(i * -(math.log(10000.0) / dim))})")
    plt.legend()

plt.tight_layout()
plt.show()
```

In the resulting plots, each dimension pair (sine and cosine) has a distinct frequency and phase. These unique combinations of frequencies and phases capture different aspects of positional relationships, making the model more robust and flexible.