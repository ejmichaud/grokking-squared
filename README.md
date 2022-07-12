# Grokking followup (eric)

I am interested in gaining a better mechanistic understanding of how transformers generalize on algorithmic datasets. We have observed that structured representations seem to be important, but I'd like to fully understand how the network uses this structure to perform computation. 

I'd also like to do some gridsearches as a more direct follow-up to our paper and also to investigate whether slingshots are actually important.

The script `scripts/train.py` is a really nice, quite general script for running experiments. It uses the OpenAI transformer implementation, with a ton of hyperparameter options, and also allows for multiple operations to be trained on concurrently. It tracks model performance on each of these sub-tasks throughout training. To see configuration options, run:
```
python scripts/train.py print_config
```
To perform a run closest to what Ouail used to make the PCA plots, run:
```
python train.py -F ../test000 run with decoder_lr=0.0002 decoder_weight_decay=1.0 training_data_fraction=0.8 n_layers=2 n_heads=1 optimization_steps=10000 log_freq=25 dropout=0.05 embedding_lr=0.001 only_input_tokens=True seed=0
```

I've included some tests, which can be executed by running the `pytest` command (no arguments needed) from this directory.

