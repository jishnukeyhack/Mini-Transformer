# Mini-Transformer

A basic Python implementation demonstrating how a **Transformer model predicts the next word using self-attention**.

This project is designed for **educational purposes** to help understand the internal working of Transformer-based language models such as **GPT, BERT, and modern LLMs**.

The model learns to generate text **word-by-word** by predicting the next token based on context.

---

## Features

- Transformer architecture implemented using **PyTorch**
- Multi-head **Self-Attention mechanism**
- **Positional Encoding**
- Transformer **Decoder blocks**
- **Next word prediction**
- **Auto-regressive text generation**
- Clean and readable code for learning

---

## How Transformers Work

Transformers process text using **self-attention**, which allows each word to focus on other words in the sentence.

Attention formula:

Attention(Q,K,V) = softmax(QKᵀ / √d) V

Where:

- Q = Query
- K = Key
- V = Value
- d = dimension of attention head

This mechanism allows the model to understand relationships between words regardless of their distance in a sentence.

---

## Project Structure
Mini-Transformer/ │ ├── transformer.py       # Main transformer model ├── tokenizer.py         # Simple tokenizer ├── train.py             # Training script ├── generate.py          # Text generation ├── requirements.txt     # Dependencies └── README.md            # Project documentation
---

## Installation

Clone the repository

```bash
git clone https://github.com/jishnukeyhack/mini-transformer.git
cd mini-transformer
