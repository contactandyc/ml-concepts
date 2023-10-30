import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel

def get_sinu_pos(seq_len, dim_emb):
    pos = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).float()
    div_term = torch.exp(torch.arange(0, dim_emb, 2).float() * -(math.log(10000.0) / dim_emb))
    sinu_pos = pos * div_term
    sinu_pos = sinu_pos.expand(1, seq_len, dim_emb//2)
    sinu_pos = torch.cat([torch.sin(sinu_pos), torch.cos(sinu_pos)], dim=-1)
    return sinu_pos

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Read the CSV and error-check
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    print("File not found.")

seq_len = 16

# Tokenize reviews
tokens = tokenizer.batch_encode_plus(df['Review'].tolist(), padding='max_length', truncation=True, max_length=seq_len)
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

class SimpleTransformerWithRoPE(nn.Module):
    def __init__(self, dim_emb, num_heads, num_labels):
        super(SimpleTransformerWithRoPE, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.attention = nn.MultiheadAttention(dim_emb, num_heads)
        self.classifier = nn.Linear(dim_emb, num_labels)  # Classifier layer

    def forward(self, input_ids, attention_mask, sinu_pos):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # The actual embeddings
        q, k = apply_rotary_pos_emb(x, x, sinu_pos)
        x, _ = self.attention(q, k, x)
        x = self.classifier(x)
        return x

# Initialize model and positional encoding
dim_emb = 768
num_heads = 2
num_labels = 3  # Number of classes (Positive, Neutral, Negative)

model = SimpleTransformerWithRoPE(dim_emb, num_heads, num_labels)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch  # Renamed input_emb to input_ids for clarity

        # Compute sinu_pos for the current batch and sequence length
        sinu_pos = get_sinu_pos(seq_len, dim_emb)
        sinu_pos = sinu_pos.expand(input_ids.size(0), -1, -1)  # Match batch size

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, sinu_pos=sinu_pos)

        outputs_cls = outputs[:, 0, :]
        loss = criterion(outputs_cls, labels)
        # loss = criterion(outputs.view(-1, num_labels), labels)
        # loss = criterion(outputs.squeeze(0), labels)  # This line may also need adjustment based on output shape
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Training Loss: {loss.item()}")
