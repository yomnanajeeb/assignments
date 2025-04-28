import numpy as np

data = [
    ("deep learning models work", "work"),
    ("artificial intelligence changes everything", "everything"),
    ("machine learning is powerful", "powerful"),
]

def tokenize(sentence):
    return sentence.lower().split()

all_words = set()
for sentence, _ in data:
    all_words.update(tokenize(sentence))
word_to_idx = {word: idx for idx, word in enumerate(sorted(all_words))}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(word_to_idx)

np.random.seed(0)
embedding_dim = 4
hidden_dim = 4
lr = 0.01

embedding = np.random.randn(vocab_size, embedding_dim)
Wxh = np.random.randn(hidden_dim, embedding_dim)
Whh = np.random.randn(hidden_dim, hidden_dim)
Why = np.random.randn(vocab_size, hidden_dim * 2)  
Wxh_b = np.random.randn(hidden_dim, embedding_dim)
Whh_b = np.random.randn(hidden_dim, hidden_dim)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy(pred, target):
    return -np.log(pred[target] + 1e-9)

for epoch in range(500):
    total_loss = 0
    for sentence, target_word in data:
        sequence = [word_to_idx[w] for w in tokenize(sentence)]
        X, y = sequence[:3], word_to_idx[target_word]

        h_forward_states = []
        h_forward = np.zeros((hidden_dim,))
        for idx in X:
            x_embed = embedding[idx]
            h_forward = np.tanh(Wxh @ x_embed + Whh @ h_forward)
            h_forward_states.append((h_forward.copy(), x_embed))

        h_backward_states = []
        h_backward = np.zeros((hidden_dim,))
        for idx in reversed(X):
            x_embed = embedding[idx]
            h_backward = np.tanh(Wxh_b @ x_embed + Whh_b @ h_backward)
            h_backward_states.insert(0, (h_backward.copy(), x_embed))  

        h_combined = np.concatenate((h_forward, h_backward))
        logits = Why @ h_combined
        probs = softmax(logits)
        loss = cross_entropy(probs, y)
        total_loss += loss

        dlogits = probs
        dlogits[y] -= 1
        dWhy = np.outer(dlogits, h_combined)

        # Backpropagation Through Time  - Forward RNN
        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dh_next = np.zeros((hidden_dim,))
        for t in reversed(range(len(X))):
            h, x_embed = h_forward_states[t]
            dtanh = (1 - h ** 2) * (dWhy[:, :hidden_dim].sum() + dh_next)
            dWxh += np.outer(dtanh, x_embed)
            dWhh += np.outer(dtanh, h_forward_states[t-1][0] if t > 0 else np.zeros_like(h))
            dh_next = Whh.T @ dtanh
            embedding[X[t]] -= lr * (Wxh.T @ dtanh)

        # Backpropagation Through Time - Backward RNN
        dWxh_b = np.zeros_like(Wxh_b)
        dWhh_b = np.zeros_like(Whh_b)
        dh_next_b = np.zeros((hidden_dim,))
        for t in range(len(X)):
            h_b, x_embed = h_backward_states[t]
            dtanh_b = (1 - h_b ** 2) * (dWhy[:, hidden_dim:].sum() + dh_next_b)
            dWxh_b += np.outer(dtanh_b, x_embed)
            dWhh_b += np.outer(dtanh_b, h_backward_states[t-1][0] if t > 0 else np.zeros_like(h_b))
            dh_next_b = Whh_b.T @ dtanh_b
            embedding[X[t]] -= lr * (Wxh_b.T @ dtanh_b)

        Why -= lr * dWhy
        Wxh -= lr * dWxh
        Whh -= lr * dWhh
        Wxh_b -= lr * dWxh_b
        Whh_b -= lr * dWhh_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("\nPrediction Phase ---")
test_sentence = "machine learning is"
test_seq = [word_to_idx[w] for w in tokenize(test_sentence)]

h_forward = np.zeros((hidden_dim,))
print("\nForward pass hidden states:")
for idx in test_seq:
    x_embed = embedding[idx]
    h_forward = np.tanh(Wxh @ x_embed + Whh @ h_forward)
    print(f"Word: {idx_to_word[idx]}, Hidden state: {h_forward}")

h_backward = np.zeros((hidden_dim,))
print("\nBackward pass hidden states:")
for idx in reversed(test_seq):
    x_embed = embedding[idx]
    h_backward = np.tanh(Wxh_b @ x_embed + Whh_b @ h_backward)
    print(f"Word: {idx_to_word[idx]}, Hidden state: {h_backward}")

h_combined = np.concatenate((h_forward, h_backward))
logits = Why @ h_combined
probs = softmax(logits)
predicted_idx = np.argmax(probs)
print("\nPredicted word:", idx_to_word[predicted_idx])
