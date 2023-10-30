Tokenization is the process of converting text into smaller chunks, like words or subwords, which are then converted to integers for the model to understand.

## Byte Pair Encoding (BPE)

Byte Pair Encoding (BPE) is an unsupervised text compression algorithm that is adapted for tokenization in NLP. The general idea is to iteratively merge frequent pairs of characters or character sequences.

Here's a simplified Python code snippet for BPE:

```python
from collections import Counter

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# Dummy vocab
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)

print(vocab)
```

## SentencePiece

SentencePiece is another unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. It's not a PyTorch package, but it's commonly used alongside PyTorch.

First, you need to install SentencePiece:

```bash
pip install sentencepiece
```

Then you can train a SentencePiece model like this:

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.Train('--input=your_text.txt --model_prefix=m --vocab_size=2000')

# Initialize and encode
sp = spm.SentencePieceProcessor()
sp.Load("m.model")

encoded = sp.EncodeAsPieces("This is a test")
print(encoded)
```

Both methods are used to create a fixed-size vocabulary and convert text into integer tokens, which is essential for training or fine-tuning GPT-3 or any other language model.