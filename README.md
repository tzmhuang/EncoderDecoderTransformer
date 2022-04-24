# CS 523 Project
This is a re-implementation of transformer architecture for machine translation based on the original paper, Attention is all you need [1].


## Run

### Download and preprocess data

To download and preprocess for WMT17 for English-German translation without byte-pair encoding, 
```
python preprocess_data.py --src_lang en --trg_lang de --data_dir ./wmt17_en_de --save_dir ./wmt17_en_de_processed.pkl --share_vocab --min_freq 2
```

To download and preprocess for WMT17 for English-German translation with byte-pair encoding,
```
python preprocess_data.py --src_lang en --trg_lang de --data_dir ./wmt17_en_de --save_dir ./wmt17_en_de_processed_bpe.pkl --share_vocab --min_freq 2 --use_bpe
```

The code would generate preprocessed data and a shared vocabluary for the dataset, and save them under the current direcrtory as `wmt17_en_de_processed.pkl` or `wmt17_en_de_processed_bpe.pkl`.

The generate data file has the following structure:
```
- wmt17_en_de_processed.pkl
    -- setting: settings used for generating data
    -- vocab: shared vocabulary
    -- valid: validation pair
    -- train: train pair
    -- test: test pair
```

## Dataset
We plan to use and updated version of the datasets used in the original paper. The datasets are:

- The standard WMT 2017 English-German dataset (https://www.statmt.org/wmt17/translation-task.html) consisting of about 4.5 million sentence pairs. 


## Metric
We use the BLEU (BiLingual Evaluation Understudy) as a metric same as the original paper. It is a n-gram based evaluation metric with score scaled between zero and one , which measures the similarity of the machine-translated text to a set of high quality reference translations. A value of 0 means that the machine-translated output has no overlap with the reference translation (low quality) while a value of 1 means there is perfect overlap with the reference translations (high quality).

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin: “Attention Is All You Need”, 2017; arXiv:1706.03762.
