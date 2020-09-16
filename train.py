import os
import torch
import random
import argparse
import pickle
import numpy as np
from utils import helper
from shutil import copyfile
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer

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
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# if you want to reproduce the result, fix the random seed
'''
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
torch.cuda.manual_seed(args.seed)
helper.print_arguments(args)
'''

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
train_batch = DataLoader('./dataset/'+args.dataset+'/train.json', args.batch_size, args, dicts)
test_batch = DataLoader('./dataset/'+args.dataset+'/test.json', args.batch_size, args, dicts)

# create the folder for saving the best models and log file
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + args.log, header="# epoch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\ttest_f1")

# create the model
trainer = GCNTrainer(args, emb_matrix=emb_matrix)

# start training
train_acc_history, train_loss_history, test_loss_history, f1_score_history = [], [], [], [0.]
test_acc_history = [0.]
for epoch in range(1, args.num_epoch+1):
    train_loss, train_acc, train_step = 0., 0., 0
    for i, batch in enumerate(train_batch):
        loss, acc = trainer.update(batch)
        train_loss += loss
        train_acc += acc
        train_step += 1
        if train_step % args.log_step == 0:
            print("train_loss: {}, train_acc: {}".format(train_loss/train_step, train_acc/train_step))

    # eval on test
    print("Evaluating on test set...")
    predictions, labels = [], []
    test_loss, test_acc, test_step = 0., 0., 0
    for i, batch in enumerate(test_batch):
        loss, acc, pred, label, _, _ = trainer.predict(batch)
        test_loss += loss
        test_acc += acc
        predictions += pred
        labels += label
        test_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    print("trian_loss: {}, test_loss: {}, train_acc: {}, test_acc: {}, f1_score: {}".format( \
        train_loss/train_step, test_loss/test_step, \
        train_acc/train_step, test_acc/test_step, \
        f1_score))

    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format( \
        epoch, train_loss/train_step, test_loss/test_step, \
        train_acc/train_step, test_acc/test_step, \
        f1_score))

    train_acc_history.append(train_acc/train_step)
    train_loss_history.append(train_loss/train_step)
    test_loss_history.append(test_loss/test_step)

    # save best model
    if epoch == 1 or test_acc/test_step > max(test_acc_history):
        trainer.save(model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"\
            .format(epoch, train_loss/train_step, test_loss/test_step, \
            train_acc/train_step, test_acc/test_step, \
            f1_score))
    test_acc_history.append(test_acc/test_step)
    f1_score_history.append(f1_score)
    print("")

print("Training ended with {} epochs.".format(epoch))
bt_test_acc = max(test_acc_history)
bt_f1_score = f1_score_history[test_acc_history.index(bt_test_acc)]
print("best test_acc/f1_score: {}/{}".format(bt_test_acc, bt_f1_score))
                                                                                           
