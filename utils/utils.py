# Copyright 2019 SanghunYun, Korea University.
# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# 
# This file has been modified by SanghunYun, Korea University
# for add fucntion of _get_device and class of output_logging.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import csv
import random
import logging

import numpy as np
import torch


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def simple_pad(input_ids, input_mask, num_tokens):
    batch_size = input_ids.size(0)

    max_count, max_index = num_tokens.max(0)
    max_count = int(max_count)
    max_index = int(max_index)

    for i in range(0, batch_size):
        i_count = int(num_tokens[i])

        if i_count < max_count:
            first = input_ids[i][0:i_count-1]
            second = torch.tensor([1] * (max_count - i_count)).cuda()
            third = input_ids[max_index][max_count-1:128]
            combined = torch.cat((first, second, third), 0)
            input_ids[i] = combined
            input_mask[i] = input_mask[max_index]

def pad_for_word_mixup(input_ids, input_mask, num_tokens, idx):
    batch_size = input_ids.size(0)
    c_input_ids = input_ids.clone()

    for i in range(0, batch_size):
        j = idx[i]
        i_count = int(num_tokens[i])
        j_count = int(num_tokens[j])

        if i_count < j_count:
            small = i
            big = j
            small_count = i_count
            big_count = j_count
            small_ids = input_ids
            big_ids = c_input_ids
        elif i_count > j_count:
            small = j
            big = i
            small_count = j_count
            big_count = i_count
            small_ids = c_input_ids
            big_ids = input_ids

        if i_count != j_count:
            first = small_ids[small][0:small_count-1]
            second = torch.tensor([1] * (big_count - small_count)).cuda()
            third = big_ids[big][big_count-1:128]
            combined = torch.cat((first, second, third), 0)
            small_ids[small] = combined
            if i_count < j_count:
                input_mask[i] = input_mask[j]


    return input_ids, c_input_ids

def mixup_op(input, l, idx):
    input_a, input_b = input, input[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    return mixed_input

def bin_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def multi_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def torch_device_one():
    return torch.tensor(1.).to(_get_device())

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def _get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words)-1)
    return vocab_words[i]

def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    #streamHandler = logging.StreamHandler()
    #streamHandler.setFormatter(fomatter)
    #logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    return logger


class output_logging(object):
    def __init__(self, mode, real_time=False, dump_dir=None):
        self.mode = mode
        self.real_time = real_time
        self.dump_dir = dump_dir if dump_dir else None

        if dump_dir:
            self.dump = open(os.path.join(dump_dir, 'logs/output.tsv'), 'w', encoding='utf-8', newline='')
            self.wr = csv.writer(self.dump, delimiter='\t')

            # header
            if mode == 'eval':
                self.wr.writerow(['Ground_truth', 'Predcit', 'sentence'])
            elif mode == 'test':
                self.wr.writerow(['Predict', 'sentence'])

    def __del__(self):
        if self.dump_dir:
            self.dump.close()

    def logs(self, sentence, pred, ground_turth=None):
        if self.real_time:
            if self.mode == 'eval':
                for p, g, s in zip(pred, ground_turth, sentence):
                    print('Ground_truth | Predict')
                    print(int(g), '         ', int(p))
                    print(s, end='\n\n')
            elif self.mode == 'test':
                for p, s in zip(pred, sentence):
                    print('predict : ', int(p))
                    print(s, end='\n\n')
        
        if self.dump_dir:
            if self.mode == 'eval':
                for p, g, s in zip(pred, ground_turth, sentence):
                    self.wr.writerow([int(p), int(g), s])
            elif self.mode == 'test':
                for p, s in zip(pred, sentence):
                    self.wr.writerow([int(p), s])

class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
