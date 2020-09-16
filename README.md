# Convolution over Dependency Tree (CDT)

Dataset and code for our EMNLP 2019 paper "Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree" [[pdf]](graph_convolutional_networks_for_sentiment_analysis_.pdf)

## Requirement

- Python 3.6.7
- PyTorch 1.2.0
- NumPy 1.17.2
- GloVe pre-trained word vectors:
  - Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors).
  - Extract the [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) to the `dataset/glove/` folder.

## Usage

Training the model:

```bash
python train.py --dataset [dataset]
```

Prepare the vocabulary files for the dataset:

```bash
python prepare_vocab.py --dataset [dataset]
```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{DBLP:conf/emnlp/SunZMML19,
  author    = {Kai Sun and
               Richong Zhang and
               Samuel Mensah and
               Yongyi Mao and
               Xudong Liu},
  editor    = {Kentaro Inui and
               Jing Jiang and
               Vincent Ng and
               Xiaojun Wan},
  title     = {Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural
               Language Processing and the 9th International Joint Conference on
               Natural Language Processing, {EMNLP-IJCNLP} 2019, Hong Kong, China,
               November 3-7, 2019},
  pages     = {5678--5687},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  url       = {https://doi.org/10.18653/v1/D19-1569},
  doi       = {10.18653/v1/D19-1569},
  timestamp = {Mon, 24 Aug 2020 19:19:32 +0200},
  biburl    = {https://dblp.org/rec/conf/emnlp/SunZMML19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License

MIT
