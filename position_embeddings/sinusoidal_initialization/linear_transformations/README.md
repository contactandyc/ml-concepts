# Linear Transformations

One attractive property of sine and cosine functions is that they can be recovered through linear transformations. Specifically, any offset \( k \) applied to a sine or cosine wave can be represented as a linear function of the original sine and cosine. This helps the model to easily learn relative positions.

![Sine Cosine image](../../images/linear_transformations_sine_cosine.png)

Let's demonstrate this using some PyTorch code:

```python
import torch
import math
import matplotlib.pyplot as plt

# Define a sequence length and a frequency
seq_len = 100
freq = 2 * math.pi / seq_len  # one full period over the sequence

# Generate a tensor of shape (seq_len), filled with a sine wave
A = torch.linspace(0, 2 * math.pi, seq_len)
sin_A = torch.sin(freq * A)

# Generate a tensor of shape (seq_len), filled with a cosine wave
cos_A = torch.cos(freq * A)

# Offset (shift phase by B)
B = math.pi / 4  # 45 degrees phase shift

# Apply the linear transformation to recover sin(A + B) and cos(A + B)
sin_A_plus_B = sin_A * torch.cos(B) + cos_A * torch.sin(B)
cos_A_plus_B = cos_A * torch.cos(B) - sin_A * torch.sin(B)

# Plot the original and shifted sine and cosine waves
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.title('Sine waves')
plt.plot(A, sin_A, label='sin(A)')
plt.plot(A, sin_A_plus_B, label='sin(A + B) through linear transformation')
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Cosine waves')
plt.plot(A, cos_A, label='cos(A)')
plt.plot(A, cos_A_plus_B, label='cos(A + B) through linear transformation')
plt.legend()

plt.tight_layout()
plt.show()
```

In this example, we have the original sine and cosine waves (`sin_A` and `cos_A`), and we apply a phase shift (`B`). Using the original waves and the phase shift, we can recreate `sin(A + B)` and `cos(A + B)` through linear transformations. The plots should show how the transformed waves closely match the actual shifted waves, demonstrating the linearity property of sine and cosine in encoding positions.
