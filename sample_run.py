import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import numpy as np
import time
import pdb

import models
import train
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one
from utils import optim, configuration, checkpoint, Bar, AverageMeter
from copy import deepcopy

from tqdm import tqdm
from tensorboardX import SummaryWriter

class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

cfg = {
  "seed": 42,
  "lr": 1e-5,
  "warmup": 0.1,
  "do_lower_case": True,
  "mode": "train_eval",
  "uda_mode": True,

  "total_steps": 10000,
  "max_seq_length": 128,
  "train_batch_size": 4,
  "eval_batch_size": 8,

  "unsup_ratio": 1,
  "uda_coeff": 1,
  "tsa": "linear_schedule",
  "uda_softmax_temp": 0.85,
  "uda_confidence_thresh": 0.45,

  "alpha": 0.75,
  "lambda_u": 75,
  "T": 0.5,
  "ema_decay": 0.999,
  "val_iteration": 1024,
  "epochs": 1024,

  "data_parallel": True,
  "need_prepro": False,
  "sup_data_dir": "data/imdb_sup_train.txt",
  "unsup_data_dir": "data/imdb_unsup_train.txt",
  "eval_data_dir": "data/imdb_sup_test.txt",

  "model_file": None,
  "pretrain_file": "BERT_Base_Uncased/bert_model.ckpt",
  "vocab": "BERT_Base_Uncased/vocab.txt",
  "task": "imdb",

  "save_steps": 100,
  "check_steps": 250,
  "results_dir": "results",

  "is_position": False
}

cfg = AttributeDict(cfg)

model_cfg = {
	"dim": 768,
	"dim_ff": 3072,
	"n_layers": 12,
	"p_drop_attn": 0.1,
	"n_heads": 12,
	"p_drop_hidden": 0.1,
	"max_len": 512,
	"n_segments": 2,
	"vocab_size": 30522
}

model_cfg = AttributeDict(model_cfg)

def get_acc(model, batch):
    # input_ids, segment_ids, input_mask, label_id, sentence = batch
    input_ids, segment_ids, input_mask, label_id = batch
    logits = model(input_ids, segment_ids, input_mask)
    _, label_pred = logits.max(1)

    result = (label_pred == label_id).float()
    accuracy = result.mean()
    # output_dump.logs(sentence, label_pred, label_id)    # output dump

    return accuracy, result 

def eval(evaluate, model_file, model):
    """ evaluation function """
    model.eval()
    results = []
    iter_bar = tqdm(sup_iter) if model_file \
        else tqdm(deepcopy(eval_iter))
    for batch in iter_bar:
        batch = [t.to(device) for t in batch]

        with torch.no_grad():
            accuracy, result = evaluate(model, batch)
        results.append(result)

        iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
    return results   


def linear_rampup(current, rampup_length=cfg.total_steps):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, current_step):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, cfg.lambda_u * linear_rampup(current_step)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha

        params = list(model.state_dict().values())
        ema_params = list(ema_model.state_dict().values())

        self.params = list(map(lambda x: x.float(), params))
        self.ema_params = list(map(lambda x: x.float(), ema_params))
        self.wd = 0.02 * cfg.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def create_model(ema=False):
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))
    model = model.to(device)

    if cfg.data_parallel:
      model = nn.DataParallel(model)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def get_acc(model, batch):
    # input_ids, segment_ids, input_mask, label_id, sentence = batch
    input_ids, segment_ids, input_mask, label_id = batch
    logits = model(input_ids, segment_ids, input_mask)
    _, label_pred = logits.max(1)

    result = (label_pred == label_id).float()
    accuracy = result.mean()
    # output_dump.logs(sentence, label_pred, label_id)    # output dump

    return accuracy, result

set_seeds(cfg.seed)

# Load Data & Create Criterion
data = load_data(cfg)

data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
    else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval

device = get_device()

model = create_model()
ema_model = create_model(ema=True)

cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

train_criterion = SemiLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

ema_optimizer= WeightEMA(model, ema_model, alpha=cfg.ema_decay)

global_step = 0
loss_sum = 0.
max_acc = [0., 0]   # acc, step

sup_batch_size = None
unsup_batch_size = None

model.train()

sup_iter = repeat_dataloader(data_iter[0])
unsup_iter = repeat_dataloader(data_iter[1])
eval_iter = data_iter[2]

iter_bar = tqdm(unsup_iter, total=cfg.total_steps)

def eval(evaluate, model_file, model):
    """ evaluation function """

    results = []
    iter_bar = tqdm(deepcopy(eval_iter))
    for batch in iter_bar:
        batch = [t.to(device) for t in batch]

        with torch.no_grad():
            accuracy, result = evaluate(model, batch)
        results.append(result)

        iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
    return results



for i, batch in enumerate(iter_bar):

  sup_batch = [t.to(device) for t in next(sup_iter)]
  unsup_batch = [t.to(device) for t in next(unsup_iter)]

  unsup_batch_size = unsup_batch_size or unsup_batch[0].shape[0]

  if unsup_batch[0].shape[0] != unsup_batch_size:
    unsup_batch = [t.to(device) for t in next(unsup_iter)]


  input_ids, segment_ids, input_mask, label_ids = sup_batch

  ori_input_ids, ori_segment_ids, ori_input_mask, \
  aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

  batch_size = input_ids.size(0)

  # Transform label to one-hot
  label_ids = torch.zeros(batch_size, 2).scatter_(1, label_ids.cpu().view(-1,1), 1).cuda()

  with torch.no_grad():
    # compute guessed labels of unlabel samples
    outputs_u = model(input_ids=ori_input_ids, segment_ids=ori_segment_ids, input_mask=ori_input_mask)
    outputs_u2 = model(input_ids=aug_input_ids, segment_ids=aug_segment_ids, input_mask=aug_input_mask)
    p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
    pt = p**(1/cfg.T)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    targets_u = targets_u.detach()
  
  concat_input_ids = [input_ids, ori_input_ids, aug_input_ids]
  concat_seg_ids = [segment_ids, ori_segment_ids, aug_segment_ids]
  concat_input_mask = [input_mask, ori_input_mask, aug_input_mask]
  concat_targets = [label_ids, targets_u, targets_u]

  # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
  int_input_ids = interleave(concat_input_ids, batch_size)
  int_seg_ids = interleave(concat_seg_ids, batch_size)
  int_input_mask = interleave(concat_input_mask, batch_size)
  int_targets = interleave(concat_targets, batch_size)

  h_zero = model(
      input_ids=int_input_ids[0],
      segment_ids=int_seg_ids[0],
      input_mask=int_input_mask[0], 
      output_h=True
  )

  h_one = model(
      input_ids=int_input_ids[1],
      segment_ids=int_seg_ids[1],
      input_mask=int_input_mask[1], 
      output_h=True
  )

  h_two = model(
      input_ids=int_input_ids[2],
      segment_ids=int_seg_ids[2],
      input_mask=int_input_mask[2], 
      output_h=True
  )

  int_h = torch.cat([h_zero, h_one, h_two], dim=0)
  int_targets = torch.cat([int_targets[0], int_targets[1], int_targets[2]])

  l = np.random.beta(cfg.alpha, cfg.alpha)
  l = max(l, 1-l)

  idx = torch.randperm(int_h.size(0))

  h_a, h_b = int_h, int_h[idx]
  target_a, target_b = int_targets, int_targets[idx]

  mixed_int_h = l * h_a + (1 - l) * h_b
  mixed_int_target = l * target_a + (1 - l) * target_b

  mixed_int_h = list(torch.split(mixed_int_h, batch_size))
  mixed_int_targets = list(torch.split(mixed_int_target, batch_size))

  logits_one = model(input_h=mixed_int_h[0])
  logits_two = model(input_h=mixed_int_h[1])
  logits_three = model(input_h=mixed_int_h[2])

  logits = [logits_one, logits_two, logits_three]


  # put interleaved samples back
  logits = interleave(logits, batch_size)
  targets = interleave(mixed_int_targets, batch_size)


  logits_x = logits[0]
  logits_u = torch.cat(logits[1:], dim=0)

  targets_x = targets[0]
  targets_u = torch.cat(targets[1:], dim=0)

  #Lx, Lu, w = train_criterion(logits_x, targets_x, logits_u, targets_u, epoch+batch_idx/cfg.val_iteration)
  Lx, Lu, w = train_criterion(logits_x, targets_x, logits_u, targets_u, global_step)


  loss = Lx + w * Lu

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ema_optimizer.step()
  
  # print loss
  global_step += 1
  loss_sum += loss.item()

  iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f'\
          % (loss.item(), Lu.item(), Lx.item()))
  
  if global_step % cfg.check_steps == 0 and global_step > 999:
    results = eval(get_acc, None, ema_model)
    total_accuracy = torch.cat(results).mean().item()
    if max_acc[0] < total_accuracy:
      max_acc = total_accuracy, global_step
    print('Accuracy : %5.3f' % total_accuracy)
    print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' %(max_acc[0], max_acc[1], global_step), end='\n\n')

