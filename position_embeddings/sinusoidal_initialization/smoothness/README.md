# Smoothness

Sine and Cosine are smooth and differentiable, making them well-suited for backpropagation in neural networks.

By using both sine and cosine, the model has two different 'channels' to capture positional information, which allows it to better learn the dependencies between different positions in the input sequence.

The smoothness and differentiability of sine and cosine functions can be illustrated by plotting the functions themselves along with their derivatives.

In PyTorch, you can compute the derivative using automatic differentiation. Here's a simple example:

```python
import torch
import matplotlib.pyplot as plt

# Create a tensor representing the values for which we want to compute the sine, cosine and their derivatives
# We set requires_grad=True to indicate that we will want to compute gradients (derivatives) later.
x = torch.linspace(-2 * 3.14159, 2 * 3.14159, 100, requires_grad=True)

# Compute sine and cosine of x
sin_x = torch.sin(x)
cos_x = torch.cos(x)

# Compute the gradients (derivatives)
sin_x.sum().backward(retain_graph=True)  # Sum needed as backward() expects a scalar
sin_x_grad = x.grad
x.grad.zero_()  # Clear the gradient before next backward pass

cos_x.sum().backward()
cos_x_grad = x.grad

# Convert to numpy for plotting
x_np = x.detach().numpy()
sin_x_np = sin_x.detach().numpy()
cos_x_np = cos_x.detach().numpy()
sin_x_grad_np = sin_x_grad.detach().numpy()
cos_x_grad_np = cos_x_grad.detach().numpy()

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.title('Sine function and its derivative')
plt.plot(x_np, sin_x_np, label='sin(x)')
plt.plot(x_np, cos_x_grad_np, label="sin'(x) (derivative)")
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Cosine function and its derivative')
plt.plot(x_np, cos_x_np, label='cos(x)')
plt.plot(x_np, -sin_x_grad_np, label="cos'(x) (derivative)")
plt.legend()

plt.tight_layout()
plt.show()
```

In these plots, you can see that the derivative of \( \sin(x) \) is \( \cos(x) \) and the derivative of \( \cos(x) \) is \( -\sin(x) \). Both the functions and their derivatives are smooth and continuous across the range, making them well-suited for optimization methods like gradient descent that rely on derivatives.
