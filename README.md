# Convolution over Dependency Tree (CDT)

Dataset and code for the paper: **Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree**. Kai Sun, [Richong Zhang](http://act.buaa.edu.cn/zhangrc/), Samuel Mensah, Yongyi Mao, Xudong Liu. EMNLP 2019. [[pdf]](graph_convolutional_networks_for_sentiment_analysis_.pdf)

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

Prepare vocabulary files for the dataset:

```bash
python prepare_vocab.py --dataset [dataset]
```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{Sun2019CDT,
  author    = {Kai Sun and
               Richong Zhang and
               Samuel Mensah and
               Yongyi Mao and
               Xudong Liu},
  title     = {Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural
               Language Processing and the 9th International Joint Conference on
               Natural Language Processing, {EMNLP-IJCNLP} 2019, Hong Kong, China,
               November 3-7, 2019},
  pages     = {5678--5687},
  year      = {2019}
}
```

## License

MIT
