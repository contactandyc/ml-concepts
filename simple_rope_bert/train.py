import torch
import torch.nn as nn
import math
from torch.optim import Adam

from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

context_window_size = 16

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Read the CSV
df = pd.read_csv('reviews.csv')

# Tokenize reviews
tokens = tokenizer.batch_encode_plus(df['Review'].tolist(), padding='max_length', truncation=True, max_length=16)
input_ids = torch.tensor(tokens['input_ids'])
attention_mask = torch.tensor(tokens['attention_mask'])

# Encode labels
df['Label'] = df['Label'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0})
labels = torch.tensor(df['Label'].to_numpy())

# Split data
train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2)
train_input_ids, train_attention_mask, train_labels = input_ids[train_idx], attention_mask[train_idx], labels[train_idx]
test_input_ids, test_attention_mask, test_labels = input_ids[test_idx], attention_mask[test_idx], labels[test_idx]

# Create DataLoader
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# Rotary Positional Encoding function
def rotate_half(x):
    return torch.cat([-x[..., 1:], x[..., :1]], dim=-1)

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = sinu_pos[:, :q.size(1), :]
    cos = rotate_half(sinu_pos)
    sin = sinu_pos - cos
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

# Simple Transformer with RoPE
class SimpleTransformerWithRoPE(nn.Module):
    def __init__(self, dim_emb, num_heads):
        super(SimpleTransformerWithRoPE, self).__init__()
        self.attention = nn.MultiheadAttention(dim_emb, num_heads)

    def forward(self, x, sinu_pos):
        q, k = apply_rotary_pos_emb(x, x, sinu_pos)
        x, _ = self.attention(q, k, x)
        return x

# Initialize BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# context_window_size = 16


# Initialize model and positional encoding
dim_emb = 768  # should match the tokenizer's model dimension
num_heads = 2

model = SimpleTransformerWithRoPE(dim_emb, num_heads)
criterion = nn.CrossEntropyLoss()  # Assuming a classification problem
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:  # Assume batch contains input_emb, sinu_pos, and labels
        input_emb, sinu_pos, labels = batch

        # Forward pass
        outputs = model(input_emb, sinu_pos)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            input_emb, sinu_pos, labels = batch
            outputs = model(input_emb, sinu_pos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}")
