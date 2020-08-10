#!/bin/sh

python extract_vocab.py -d "dataset/data" -o "dataset"
python write_vocab_embedding.py -v "dataset/vocab.json" -o "dataset"

python main.py -c "dataset/data" -t "dataset/data" -o "output/output" -b 64 -e 2 -v "dataset/vocab.json" -m "dataset/vocab_emb.npy" -n 4 -l 0 -r 0.001

python precompute_embeddings.py -d "dataset/data" -m "output/output.ep0" -p "train"
python precompute_embeddings.py -d "dataset/data" -m "output/output.ep0" -p "test"
