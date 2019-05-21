import pickle

with open('vocab.pickle', 'rb') as f:
    vocab_l = pickle.load(f)

print(vocab_l)