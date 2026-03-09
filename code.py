import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Hyperparameters
# -----------------------------

vocab_size = 8000
embed_dim = 128
num_heads = 8
num_layers = 4
ff_dim = 512
max_len = 64
dropout = 0.1


# -----------------------------
# Simple Tokenizer
# -----------------------------

class SimpleTokenizer:

    def __init__(self):

        self.word2id = {}
        self.id2word = {}

    def fit(self, text):

        words = text.split()

        unique = list(set(words))

        for i, w in enumerate(unique):

            self.word2id[w] = i
            self.id2word[i] = w

    def encode(self, text):

        return [self.word2id[w] for w in text.split() if w in self.word2id]

    def decode(self, tokens):

        return " ".join([self.id2word[t] for t in tokens])


# -----------------------------
# Positional Encoding
# -----------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)].to(x.device)


# -----------------------------
# Multi Head Attention
# -----------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self, dim, heads):

        super().__init__()

        self.heads = heads

        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, x):

        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T)).to(device)

        scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)

        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


# -----------------------------
# Feed Forward
# -----------------------------

class FeedForward(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, x):

        return self.net(x)


# -----------------------------
# Transformer Block
# -----------------------------

class Block(nn.Module):

    def __init__(self):

        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads)

        self.ff = FeedForward(embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))

        x = x + self.ff(self.norm2(x))

        return x


# -----------------------------
# GPT Model
# -----------------------------

class MiniGPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos = PositionalEncoding(embed_dim, max_len)

        self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        x = self.pos(x)

        x = self.blocks(x)

        x = self.norm(x)

        logits = self.head(x)

        return logits


# -----------------------------
# Training Data
# -----------------------------

text = """
machine learning is amazing
deep learning builds powerful models
transformers use attention mechanism
ai can generate answers
"""

tokenizer = SimpleTokenizer()

tokenizer.fit(text)

tokens = tokenizer.encode(text)

data = torch.tensor(tokens)


# -----------------------------
# Training
# -----------------------------

model = MiniGPT().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for step in range(500):

    ix = torch.randint(0, len(data) - max_len - 1, (1,))

    x = data[ix:ix + max_len].unsqueeze(0).to(device)

    y = data[ix + 1:ix + max_len + 1].unsqueeze(0).to(device)

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:

        print("loss:", loss.item())


# -----------------------------
# Text Generation
# -----------------------------

def generate(prompt, steps=20):

    tokens = tokenizer.encode(prompt)

    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(steps):

        logits = model(tokens)

        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(tokens[0].tolist())


print(generate("machine learning"))
