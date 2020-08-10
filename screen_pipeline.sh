#!/bin/sh



# move precomputed stuff
mv UI_embedding/trainscreen_names.json precomp
mv UI_embedding/trainui_emb.json precomp
mv UI_embedding/trainuis.json precomp
mv UI_embedding/traindsc_emb.npy precomp
mv UI_embedding/traindescr.json precomp

mv UI_embedding/testscreen_names.json precomp
mv UI_embedding/testui_emb.json precomp
mv UI_embedding/testuis.json precomp
mv UI_embedding/testdsc_emb.npy precomp
mv UI_embedding/testdescr.json precomp

# train the layout model

python layout.py -d "UI_embedding/dataset/screens_small" -b 32 -e 4 -r 0.001

# precompute layout embeddings
python write_layout.py -d "UI_embedding/dataset/data" -m "output/autoencoder.ep4" -p "precomp/train"
python write_layout.py -d "UI_embedding/dataset/data" -m "output/autoencoder.ep4" -p "precomp/test"

python main_preloaded.py -c "precomp/train" -t "precomp/test" -o "output/model" -b 256 -e 200 -n 4 -r 0.001 -s 128 -v 5

