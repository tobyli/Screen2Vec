# Screen2Vec

This repository houses the files necessary to implement the Screen2Vec vector embedding process on screens from the [RICO dataset](http://interactionmining.org/rico).


## Steps to run full pipeline

1. Extract vocab (`UI_embedding/extract_vocab.py`) from the full RICO traces dataset and then pre-write its Sentence-BERT embeddings (`UI_embedding/write_vocab_embedding.py`).
2. Train UI embedding model (UI_embedding/main.py) on RICO traces dataset.
3. Use `UI_embedding/modeltester.py` to test the performance.
4. Use `UI_embedding/precompute_embeddings` to write files of the UI components and app descriptions and their respective embeddings; do this for both the test and train components of the dataset.
5. Train the layout autoencoder model using the RICO screens dataset using `layout.py` and then precompute the layout embeddings for your dataset using `write_layout.py`.
6. Train the Screen2Vec embedding model using `main_preloaded.py`.
7. Use `modeltester_screen.py` to test it out.

## In-depth
-------------

## Setup

The code was developed in the following environment:

- Python 3.6.1
- Pytorch 1.5.0

To install dependencies:

- Python dependencies
  ```
  pip install -r requirements.txt
  ```

## Data

Due to its large size, the data is hosted outside Github: the data is stored at <http://interactionmining.org/rico> in the interaction traces dataset. To easily download it, run
```
./download_data.sh
```

The pretrained models are stored at <here>. Download them by running
```
./download_models.sh
```
The model labelled "UI2Vec" is the GUI element embedding model, "Screen2Vec" is the screen embedding model, "layout_encoder" is the screen layout embedder, and "visual_encoder" is our visual autoencoder baseline.


## Quick start

If you just want to embed a screen using our pretrained GUI element, layout, and screen embedding models, run:

```
python get_embedding.py -s <path-to-screen> -u "UI_embedding/output/slow_uichange.ep120" -m "output/final_4.ep120" -l "output/autoencoder.ep800" -v 4

```

- This generates the vector embedding using our pretrained models


## Training

There are 2-3 steps for training your own screen embedding- first training the GUI element model, optionally training the screen layout autoencoder, and then the screen level model

### GUI model

Files for the GUI model are contained within the UI_embedding directory.
To train a model, run from within that directory:

```
python main.py -d <location-of-dataset> -o <desired-output-path> -b 256 -e 100 -v "vocab.json" -m "vocab_emb.npy" -n 16 -r 0.001 -l "cel"
```

Then, to pre-generate these embeddings for your dataset to then use in screen training, run:

```
python precompute_embeddings.py -d <location-of-dataset> -m <desired-ui-model> -p <desired-prefix> -n 16
```

### Layout autoencoder

The autoencoder is trained from within the Screen2Vec directory, by running:

```
python layout.py -d <location-of-screen-dataset> -b 256 -e 400 -r 0.001
```
where -b flags the batch size, -e the number of epochs, and -r the learning rate. Here, use the screen dataset rather than the trace dataset.

Then, to pre-generate these embeddings for your dataset to then use in screen training, run

```
python write_layout.py -d <location-of-dataset> -m <layout-model> -p <same-desired-prefix>
```

### Screen model

To train the Screen level model, run

```
python main_preloaded.py -d <previously-chosen-prefix> -s 128 -b 256 -t <ui-output-prefix>
```


## Evaluation

There are files to evaluate the performance of both the GUI and Screen embeddings

To test the prediction accuracy of the GUI embedding, run (from within the UI_embedding directory)

```
python modeltester.py
```

To test the prediction accuracy of the Screen embedding, run

```
python modeltester_scre
```


## Reference

> 
> 
> 


