from sentence_transformers import SentenceTransformer
import json
import numpy as np

with open("dataset/vocab.json") as f:
    vocab_list = json.load(f, encoding='utf-8')

bert = SentenceTransformer('bert-base-nli-mean-tokens')

vocab_emb = bert.encode(vocab_list)

np.save("vocab_emb", vocab_emb)