from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Read the CSV
df = pd.read_csv('reviews.csv')

# Tokenize reviews
tokens = tokenizer.batch_encode_plus(df['Review'].tolist(), padding=True, truncation=True, max_length=512)
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

# Now you can use train_loader and test_loader in your training and evaluation loops.
# Loop through train_loader and print results
print("Train DataLoader:")
for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}")
    print("Input IDs:", input_ids)
    for i, ids in enumerate(input_ids):
        decoded_text = tokenizer.decode(ids.tolist())
        print(f"Decoded Text {i+1}: {decoded_text}")
    print("Attention Mask:", attention_mask)
    print("Labels:", labels)
    print("------")

# Loop through test_loader and print results
print("Test DataLoader:")
for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_loader):
    print(f"Batch {batch_idx+1}")
    print("Input IDs:", input_ids)
    for i, ids in enumerate(input_ids):
        decoded_text = tokenizer.decode(ids.tolist())
        print(f"Decoded Text {i+1}: {decoded_text}")
    print("Attention Mask:", attention_mask)
    print("Labels:", labels)
    print("------")
