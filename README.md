# CS 523 Project
This is a re-implementation of transformer architecture for machine translation based on the original paper, Attention is all you need [1].

## Dataset
We plan to use datasets identical to that used in the original paper. The datasets are:

- The standard WMT 2014 English-German dataset (https://nlp.stanford.edu/projects/nmt/) consisting of about 4.5 million sentence pairs. 
- A larger WMT 2014 English-French dataset consisting of 36M sentences.


## Metric
We use the BLEU (BiLingual Evaluation Understudy) as a metric same as the original paper. It is a n-gram based evaluation metric with score scaled between zero and one , which measures the similarity of the machine-translated text to a set of high quality reference translations. A value of 0 means that the machine-translated output has no overlap with the reference translation (low quality) while a value of 1 means there is perfect overlap with the reference translations (high quality).

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin: “Attention Is All You Need”, 2017; arXiv:1706.03762.
