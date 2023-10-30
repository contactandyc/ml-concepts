# Periodic Nature

The periodicity of sine and cosine functions allows machine learning models to handle sequences of varying lengths more effectively. In natural language processing tasks, where sentences can differ greatly in length, this periodicity helps the model generalize better. Specifically, positions that are a full cycle apart—like 0 and 2π—receive similar encodings. This consistency aids the model in recognizing patterns and relationships between positions, irrespective of the sequence's overall length.

Here's some simple PyTorch code to illustrate the periodicity of sine and cosine functions:

```python
import torch
import matplotlib.pyplot as plt

# Generate an array of positions from 0 to 4π
positions = torch.linspace(0, 4 * 3.14159, 1000)

# Compute sine and cosine values for these positions
sin_values = torch.sin(positions)
cos_values = torch.cos(positions)

# Plot the values
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title('Sine Function')
plt.plot(positions, sin_values)
plt.xlabel('Position')
plt.ylabel('sin(Position)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title('Cosine Function')
plt.plot(positions, cos_values)
plt.xlabel('Position')
plt.ylabel('cos(Position)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

In the plot, you'll see that both the sine and cosine functions repeat their values in a regular pattern—every \(2\pi\) units. This is beneficial for the model as it learns to associate not just absolute positions but also relative spacing between different positions, allowing it to generalize better across different sequence lengths.
