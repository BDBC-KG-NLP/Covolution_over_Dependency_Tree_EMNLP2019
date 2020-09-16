"""
Prepare the constants for the datasest.
"""
import json
import argparse
import numpy as np
from collections import Counter
import pickle

VOCAB_PREFIX = ['[PAD]', '[UNK]']


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--dataset', help='TACRED directory.')
    parser.add_argument('--lower', default=True, help='If specified, lowercase all words.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train_file = './dataset/'+args.dataset+'/train.json'
    test_file = './dataset/'+args.dataset+'/test.json'

    print("loading tokens...")
    train_tokens, train_pos, train_dep, train_max_len = load_tokens(train_file)
    test_tokens, test_pos, test_dep, test_max_len = load_tokens(test_file)
    if args.lower:
        train_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
                (train_tokens, test_tokens)]

    vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
    emb_file = './dataset/'+args.dataset+'/embedding.npy'

    print("loading glove words...")
    glove_vocab = load_glove_vocab(args.wv_file, args.wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))

    print("building vocab...")
    v = build_vocab(train_tokens+test_tokens, glove_vocab)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = build_embedding(args.wv_file, v, args.wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)

    print('saving the dicts...')
    ret = dict()
    pos_list = VOCAB_PREFIX+list(set(train_pos+test_pos))
    pos_dict = {pos_list[i]:i for i in range(len(pos_list))}
    dep_list = VOCAB_PREFIX+list(set(train_dep+test_dep))
    dep_dict = {dep_list[i]:i for i in range(len(dep_list))}
    max_len = max(train_max_len, test_max_len)
    post_list = VOCAB_PREFIX+list(range(-max_len, max_len))
    post_dict = {post_list[i]:i for i in range(len(post_list))}
    ret['pos'] = pos_dict
    ret['dep'] = dep_dict
    ret['post'] = post_dict
    ret['polarity'] = {'positive':0, 'negative':1, 'neutral':2}
    open('./dataset/'+args.dataset+'/constant.py', 'w').write(str(ret))

    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        pos = []
        dep = []
        max_len = 0
        for d in data:
            tokens.extend(d['token'])
            pos.extend(d['pos'])
            dep.extend(d['deprel'])
            max_len = max(len(d['token']), max_len)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, list(set(pos)), list(set(dep)), max_len

def load_glove_vocab(filename, wv_dim):
    vocab = set()
    with open('./dataset/glove/'+filename, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)
    return vocab

def build_vocab(tokens, glove_vocab):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # sort words according to its freq
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens
    v = VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # pad vector
    w2id = {w: i for i, w in enumerate(vocab)}
    with open('./dataset/glove/'+wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched


if __name__ == '__main__':
    main()


