import os
import random
import argparse
import numpy as np
import pickle
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer
from utils import helper

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Restaurants')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=100, help='GCN mem dim.')
parser.add_argument('--rnn_hidden', type=int, default=100, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0., help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models/best_model.pt', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# load contants
dicts = eval(open('./dataset/'+args.dataset+'/constant.py', 'r').read())
vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
token_vocab = dict()
with open(vocab_file, 'rb') as infile:
    token_vocab['i2w'] = pickle.load(infile)
    token_vocab['w2i'] = {token_vocab['i2w'][i]:i for i in range(len(token_vocab['i2w']))}
emb_file = './dataset/'+args.dataset+'/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(token_vocab['i2w'])
assert emb_matrix.shape[1] == args.emb_dim

args.token_vocab_size = len(token_vocab['i2w'])
args.post_vocab_size = len(dicts['post'])
args.pos_vocab_size = len(dicts['pos'])

dicts['token'] = token_vocab['w2i']

# load training set and test set
print("Loading data from {} with batch size {}...".format(args.dataset, args.batch_size))
test_batch = DataLoader('./dataset/'+args.dataset+'/test.json', args.batch_size, args, dicts)

# create the model
trainer = GCNTrainer(args, emb_matrix=emb_matrix)

print("Loading model from {}".format(args.save_dir))
trainer.load(args.save_dir)

print("Evaluating...")
predictions, labels = [], []
test_loss, test_acc, test_step = 0., 0., 0
for i, batch in enumerate(test_batch):
    loss, acc, pred, label, _, _ = trainer.predict(batch)
    test_loss += loss
    test_acc += acc
    predictions += pred
    labels += label
    test_step += 1
f1_score = metrics.f1_score(labels, predictions, average='macro')

print("test_loss: {}, test_acc: {}, f1_score: {}".format( \
                                      test_loss/test_step, \
                                      test_acc/test_step, \
                                      f1_score))
