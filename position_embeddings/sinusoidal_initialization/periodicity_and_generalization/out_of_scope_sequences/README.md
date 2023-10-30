# Out of Scope Sequences

The periodic nature of sine and cosine functions can be extremely useful for sequence models like Transformers, which may need to generalize to sequence lengths not seen during training.

The sine and cosine functions repeat their values at regular intervals (periods). As a result, the positional encodings for positions separated by a full period (e.g., \(2\pi\)) will be similar. This means that even if the model has never seen a sequence of a particular length during training, the periodicity of the sine and cosine functions will help it make reasonable predictions for those out-of-scope sequence lengths.

Here's a simplified PyTorch code snippet to demonstrate this:

```python
import torch
import math
import matplotlib.pyplot as plt

# Initialize sequence lengths and dimensions
seq_len1, seq_len2 = 50, 100  # Two different sequence lengths
dim = 64  # Embedding dimension

# Function to generate sinusoidal positional encodings
def sinusoidal_encoding(seq_len, dim):
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
    pos_enc = pos * div_term
    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])
    return pos_enc

# Generate sinusoidal encodings for different sequence lengths
encoding1 = sinusoidal_encoding(seq_len1, dim)
encoding2 = sinusoidal_encoding(seq_len2, dim)

# Plot to demonstrate the periodicity
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title(f"Sinusoidal Positional Encoding for seq_len = {seq_len1}")
plt.imshow(encoding1.detach().numpy())
plt.colorbar()

plt.subplot(2, 1, 2)
plt.title(f"Sinusoidal Positional Encoding for seq_len = {seq_len2}")
plt.imshow(encoding2.detach().numpy())
plt.colorbar()

plt.tight_layout()
plt.show()
```

The above code will generate two heatmaps of sinusoidal positional encodings for sequences of lengths 50 and 100. You'll notice that the patterns are similar, thanks to the periodic nature of sine and cosine functions. This similarity allows the model to generalize to different sequence lengths.
