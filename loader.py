import json
import random
import torch
import numpy as np

class DataLoader(object):
    def __init__(self, filename, batch_size, args, dicts):
        self.batch_size = batch_size
        self.args = args
        self.dicts = dicts

        with open(filename) as infile:
            data = json.load(infile)
        
        # preprocess data
        data = self.preprocess(data, dicts, args)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, dicts, args):
        
        processed = []

        for d in data:
            for aspect in d['aspects']:
                # word token
                tok = list(d['token'])
                if args.lower == True:
                    tok = [t.lower() for t in tok]
                
                asp = list(aspect['term']) # aspect
                label = aspect['polarity'] # label 
                pos = list(d['pos'])       # pos
                head = list(d['head'])     # head
                length = len(tok)          # real length
                # position
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # mask of aspect
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]

                # map to ids 
                tok = map_to_ids(tok, dicts['token'])
                asp = map_to_ids(asp, dicts['token'])
                label = dicts['polarity'][label]
                pos = map_to_ids(pos, dicts['pos'])
                head = [int(x) for x in head]
                assert any([x == 0 for x in head])
                post = map_to_ids(post, dicts['post'])
                assert len(tok) == length \
                       and len(pos) == length \
                       and len(head) == length \
                       and len(post) == length \
                       and len(mask) == length

                processed += [(tok, asp, pos, head, post, mask, length, label)]

        return processed

    def gold(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        
        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        tok = get_long_tensor(batch[0], batch_size)
        asp = get_long_tensor(batch[1], batch_size)
        pos = get_long_tensor(batch[2], batch_size)
        head = get_long_tensor(batch[3], batch_size)
        post = get_long_tensor(batch[4], batch_size)
        mask = get_float_tensor(batch[5], batch_size)
        length = torch.LongTensor(batch[6])
        label = torch.LongTensor(batch[7])

        return (tok, asp, pos, head, post, mask, length, label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else 1 for t in tokens] # the id of [UNK] is ``1''
    return ids

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
