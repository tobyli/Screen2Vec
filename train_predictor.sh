#!/bin/sh

python write_baseline_models_for_prediction.py -d "/home/toby/Screen2Vec/datasets/filtered_traces" -o "output/baseline" -l "output/autoencoder.ep800"
python for_baselines.py -d "output/baselinetext_eval.json" -o "output/text_predictor" -b 256 -e 300 -n 4 -r 0.0005