#!/bin/sh



# move precomputed stuff
mv UI_embedding/fullscreen_names.json precomp
mv UI_embedding/fullui_emb.json precomp
mv UI_embedding/fulluis.json precomp
mv UI_embedding/fulldsc_emb.npy precomp
mv UI_embedding/fulldescr.json precomp


# train the layout model

python layout.py -d "UI_embedding/dataset/screens_small" -b 32 -e 4 -r 0.001

# precompute layout embeddings
python write_layout.py -d "UI_embedding/dataset/data" -m "output/autoencoder.ep4" -p "precomp/full"

python main_preloaded.py -d "precomp/full" -o "output/model" -b 256 -e 200 -n 4 -r 0.001 -s 128 -v 5

