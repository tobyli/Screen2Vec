# Screen2Vec

Screen2Vec is a new self-supervised technique for generating more comprehensive semantic embeddings of GUI screensand components using their textual content, visual designand layout patterns, and app meta-data. Learn more about Screen2Vec in our [CHI 2021 paper](http://toby.li/files/li-screen2vec-chi2021.pdf).

This repository houses the files necessary to implement the Screen2Vec vector embedding process on screens from the [RICO dataset](http://interactionmining.org/rico).


## Steps to run full pipeline

1. Extract vocab from the full RICO traces dataset and pre-write its Sentence-BERT embeddings (`UI_embedding/extract_vocab.py`).
2. Train UI embedding model (UI_embedding/main.py) on RICO traces dataset.
3. Use `UI_embedding/modeltester.py` to test the performance.
4. Use `UI_embedding/precompute_embeddings` to write files of the UI components and app descriptions and their respective embeddings; do this for both the test and train components of the dataset.
5. Train the layout autoencoder model using the RICO screens dataset using `layout.py` and then precompute the layout embeddings for your dataset using `write_layout.py`.
6. Train the Screen2Vec embedding model using `main_preloaded.py`.
7. Use `modeltester.py` to test it out.

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

Due to its large size, the data is hosted outside Github: the data is stored at <http://interactionmining.org/rico> in the "Interaction Traces" and "UI Screenshots and View Hierarchies" datasets. To easily download it, run
```
./download_data.sh
```

The pretrained models are stored at <http://basalt.amulet.cs.cmu.edu/screen2vec/>. Download them by running
```
./download_models.sh
```
The model labelled "UI2Vec" is the GUI element embedding model, "Screen2Vec" is the screen embedding model, "layout_encoder" is the screen layout autoencoder, and "visual_encoder" is our visual autoencoder baseline.


## Quick start

If you just want to embed a screen using our pretrained GUI element, layout, and screen embedding models, run:

```
python get_embedding.py -s <path-to-screen> -u "UI2Vec_model.ep120" -m "Screen2Vec_model_v4.ep120" -l "layout_encoder.ep800"

```

This generates the vector embedding using our pretrained models. The parameters are:
- -s/--screen, the path to the screen to encode
- -u/--ui_model, the path to the ui embedding model to use
- -m/--screen_model, the path to the screen embedding model to use
- -l/--layout_model, the path to the layout embedding model to use


## Training

There are 2-3 steps for training your own screen embedding- first training the GUI element model, optionally training the screen layout autoencoder, and then the screen level model

### GUI model

Files for the GUI model are contained within the UI_embedding directory.
To train a model, run from within that directory:

```
python extract_vocab.py -d <location-of-dataset>

python main.py -d <location-of-dataset> -o <desired-output-path> -b 256 -e 100 -v "vocab.json" -m "vocab_emb.npy" -n 16 -r 0.001 -l "cel"
```
The parameters here are:
- -d/--dataset, the path to the RICO dataset traces
- -o/--output, the path prefix for where the output models should be stored
- -b/--batch, number of traces in a batch
- -e/--epochs, desired number of epochs
- -v/--vocab_path, the path to where the vocab was precomputed
- -m/--embedding_path, path to where the vocab BERT embeddings were precomputed
- -n/--num_predictors, the number of UI elements used to predict the unknown element
- -r/--rate, the training rate
- -hi/--hierarchy, flag to use if desiring to use the hierarchy distance metric rather than euclidean
- -l/--loss, the desired loss metric; "cel" for cross-entropy loss, or "cossim" for cosine similarity

Then, to pre-generate these embeddings for your dataset to then use in screen training, run:

```
python precompute_embeddings.py -d <location-of-dataset> -m <desired-ui-model> -p <desired-prefix>
```
Then, move the files generated here from the UI_embedding directory into the Screen2Vec directory.

### Layout autoencoder

The autoencoder is trained from within the Screen2Vec directory, by running:

```
python layout.py -d <location-of-screen-dataset> -b 256 -e 400 -r 0.001
```
where -b flags the batch size, -e the number of epochs, and -r the learning rate. Here, use the screen dataset ("combined") rather than the trace dataset ("filtered_traces").

Then, to pre-generate these embeddings for your dataset to then use in screen training, run

```
python write_layout.py -d <location-of-dataset> -m <layout-model> -p <same-desired-prefix>
```
Here, make sure you use the same desired prefix as from precomputing the UI embeddings.

### Screen model

To train the Screen level model, run

```
python main_preloaded.py -d <previously-chosen-prefix> -s 128 -b 256 -t <ui-output-prefix>
```

The parameters here are:
- -d/--data, the prefix selected on the precompute embeddings stage
- -s/--neg_samp, the number of screens to use as a negative sample during training
- -b/--batch, number of traces in a batch
- -e/--epochs, desired number of epochs
- -n/--num_predictors, the number of screens used to predict the next screen in the trace (we used 4)
- -r/--rate, the training rate
- -t/--test_train_split, the output path prefix from the UI embedding model, which was used to store the data split information as well

## Evaluation

There are files to evaluate the performance of both the GUI and Screen embeddings

To test the prediction accuracy of the GUI embedding, run (from within the UI_embedding directory)

```
python modeltester.py -m "UI2Vec_model.ep120" -v "vocab.py" -ve "vocab_emb.npy" -d <path-to-dataset>
```
- -m/--model, path to the pretrained UI embedding model
- -d/--dataset, the path to the RICO dataset traces
- -v/--vocab_path, the path to where the vocab was precomputed
- -ve/--vocab_embedding_path, path to where the vocab BERT embeddings were precomputed

To test the prediction accuracy of the Screen embedding, run (from within the main directory)

```
python modeltester_screen.py -m "Screen2Vec_model_v4.ep120" -v 4 -n 4
```
where -m flags the model to test, -v the model version (4 is standard), and -n the number of predictors used in predictions 

## Other files

The following files were used in the testing infrastructure of our baseline models. These are not needed for general use in the Screen2Vec pipeline and therefore have been stored in the sub-directory baseline to avoid clutter/confusion. However, if you desire to run these scripts, they should be moved to the main directory:
- for_baselines.py
- modeltester_baseline.py
- write_baseline_models.py
- write_baseline_models_for_prediction.py


## Reference

> Toby Jia-Jun Li*, Lindsay Popowski*, Tom M. Mitchell, and Brad A. Myers. [Screen2Vec: Semantic Embedding of GUI Screens and GUI Components](http://toby.li/files/li-screen2vec-chi2021.pdf). Proceedings of the ACM Conference on Human Factors in Computing Systems (CHI 2021).



