from sentence_transformers import SentenceTransformer
import json

with open("vocab.json") as f:
    vocab_list = json.load(f, encoding='utf-8')

bert = SentenceTransformer('bert-base-nli-mean-tokens')

vocab_emb = bert.encode(vocab_list)

with open('emb_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_emb, f, indent=4)