# Encoder-Decoder Transformer
This is a re-implementation of transformer architecture for machine translation based on the original paper, Attention is all you need [1].


## Data Processing

The code for data processing are located in `./data` file

### Download and preprocess WMT translation data

To download and preprocess for WMT17 for English-German translation without byte-pair encoding, 
```
python preprocess_data.py --src_lang en --trg_lang de --data_dir ./wmt17_en_de --save_dir ./wmt17_en_de_processed.pkl --share_vocab --min_freq 2
```

To download and preprocess for WMT17 for English-German translation with byte-pair encoding,
```
python preprocess_data.py --src_lang en --trg_lang de --data_dir ./wmt17_en_de --save_dir ./wmt17_en_de_processed_bpe.pkl --share_vocab --min_freq 2 --use_bpe
```

The code would generate preprocessed data and a shared vocabluary for the dataset, and save them under the current direcrtory as `wmt17_en_de_processed.pkl` or `wmt17_en_de_processed_bpe.pkl`.

If byte-pair encoding is enabled, the code would also generate a vocabulary of bpe codes and save it as `./<data_dir>/bpe_code` by default.

The generate data file has the following structure:
```
- wmt17_en_de_processed.pkl
    -- setting: settings used for generating data
    -- vocab: shared vocabulary
    -- valid: validation pair
    -- train: train pair
    -- test: test pair
```

### Download and preprocess MLQE quality estimation data

The MLQE has been compiled by the community and uploaded to github at [here](https://github.com/facebookresearch/mlqe). To download the dataset, clone the repository.

```
git clone https://github.com/facebookresearch/mlqe.git
```

**Before processing MLQE data, make sure you have generated vocabulary from WMT translation corpus by running command in the previous section.**

After cloning the data repo, process data by running preprocess_qe_data.py.
To process MLQE English-German quality estimation data with previously generated vocabulary `wmt17_en_de_processed.pkl` and without BPE.

```
python preprocess_qe_data.py --src_lang en --trg_lang de --source_dir <path_to_mlqe_repo> --vocab_file wmt17_en_de_processed.pkl --save_dir ./mlqe_en_de_processed.pkl
```

To process MLQE English-German quality estimation data with previously generated vocabulary `wmt17_en_de_processed_bpe.pkl` and with BPE. 

```
python preprocess_qe_data.py --src_lang en --trg_lang de --source_dir <path_to_mlqe_repo> --vocab_file wmt17_en_de_processed_bpe.pkl --save_dir ./mlqe_en_de_processed_bpe.pkl --use_bpe --bpe_codes ./bpe_codes
```

Notice `./bpe_codes` was generated when running BPE on WMT data in the previous section.


The code would generate preprocessed data and a shared vocabluary for the dataset, and save them under the current direcrtory as `mlqe_en_de_processed.pkl` or `mlqe_en_de_processed_bpe.pkl`.

**Notice if BPE is desired for MLQE data, we should use vocabulary generated by byte-pair encoded data.**

The generate data file has the following structure:
```
- mlqe_en_de_processed.pkl
    -- vocab_setting: settings inherited from vocab_file
    -- setting: settings used for generating data
    -- vocab: shared vocabulary
    -- valid: validation pair
    -- train: train pair
    -- test: test pair
```



## Translation on WMT17 Tataset

The code for translation is located under `./MT`.

### Our Implementation

```
python train.py --data_path ./wmttest_en_de_processed_bpe.pkl --batch_size 128 --log_dir ./log
```

### Baseline

```
python train_baseline.py --data_path ./wmttest_en_de_processed_bpe.pkl --batch_size 128 --log_dir ./log
```


## Translation on Multi30k dataset

The code for translation is located under `./MT`.

### Our Implementaiton

```
python train_multi30k.py --batch_size 128 --log_dir ./log
```

### Baseline

```
python train_multi30k_baseline.py --batch_size 128 --log_dir ./log
```

## Quality Estimation

The code for our quality estimation models can be found in the files listed in the table below

| Model           | Script                   |
| --------------- | ------------------------ |
| Transformer+MLP | `QE/Transformer_QE.ipynb`   |
| BERT+MLP        | `QE/Bert_QE.ipynb`          |
| XML+MLP         | `QE/transformer_xml.ipynb`  |
| NUQE+MLP        | `QE/transformer_nuqe.ipynb` |

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin: “Attention Is All You Need”, 2017; arXiv:1706.03762.

## Code references

[1] https://jaketae.github.io/study/relative-positional-encoding/

[2] https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master

[3] https://github.com/enhuiz/transformer-pytorch

[4] https://nlp.seas.harvard.edu/2018/04/03/attention.html
