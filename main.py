# Copyright 2019 SanghunYun, Korea University.
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
import pdb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import train
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one, mixup_op, pad_for_word_mixup, simple_pad, sigmoid_rampup
from utils import optim, configuration
import numpy as np


from dataset import DataSet
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

parser = argparse.ArgumentParser(description='PyTorch UDA Training')

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--data_seed', default=42, type=int)
parser.add_argument('--lr', default=0.00004, type=float)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('--do_lower_case', default=True, type=bool)
parser.add_argument('--mode', default='train_eval', type=str)
parser.add_argument('--model_cfg', default='config/bert_base.json', type=str)
parser.add_argument('--hide_tqdm', action='store_true')

#Dataset
parser.add_argument('--task', default="imdb", type=str)
parser.add_argument('--num_labels', default=2, type=int)
parser.add_argument('--train_cap', default=-1, type=int)
parser.add_argument('--dev_cap', default=-1, type=int)
parser.add_argument('--unsup_cap', default=-1, type=int)

parser.add_argument('--uda_mode', action='store_true')
parser.add_argument('--mixmatch_mode', action='store_true')
parser.add_argument('--uda_test_mode', action='store_true')
parser.add_argument('--uda_test_mode_two', action='store_true')
parser.add_argument('--unsup_mixup', action='store_true')

parser.add_argument('--total_steps', default=10000, type=int)
parser.add_argument('--check_after', default=4999, type=int)
parser.add_argument('--early_stopping', default=10, type=int)
parser.add_argument('--max_seq_length', default=128, type=int)
parser.add_argument('--train_batch_size', default=16, type=int)
parser.add_argument('--eval_batch_size', default=16, type=int)

parser.add_argument('--no_sup_loss', action='store_true')
parser.add_argument('--no_unsup_loss', action='store_true')

#UDA
parser.add_argument('--unsup_ratio', default=1, type=int)
parser.add_argument('--uda_coeff', default=1, type=int)
parser.add_argument('--tsa', default='linear_schedule', type=str)
parser.add_argument('--uda_softmax_temp', default=0.85, type=float)
parser.add_argument('--uda_confidence_thresh', default=0.45, type=float)
parser.add_argument('--unsup_criterion', default='KL', type=str)

#MixMatch
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--lambda_u', default=75, type=int)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema_decay', default=0.999, type=float)
parser.add_argument('--sup_mixup',  choices=['cls', 'word', 'word_cls'])
parser.add_argument('--mixup', choices=['cls', 'word', 'word_cls'])
parser.add_argument('--simple_pad', action='store_true')
parser.add_argument('--manifold_mixup', action='store_true')
parser.add_argument('--consistency_rampup_starts', default=0, type=int)
parser.add_argument('--consistency_rampup_ends', default=0, type=int)

parser.add_argument('--data_parallel', default=True, type=bool)
parser.add_argument('--need_prepro', default=False, type=bool)
parser.add_argument('--sup_data_dir', default='data/imdb_sup_train.txt', type=str)
parser.add_argument('--unsup_data_dir', default="data/imdb_unsup_train.txt", type=str)
parser.add_argument('--eval_data_dir', default="data/imdb_sup_test.txt", type=str)

parser.add_argument('--model_file', default="", type=str)
parser.add_argument('--pretrain_file', default="BERT_Base_Uncased/bert_model.ckpt", type=str)
parser.add_argument('--vocab', default="BERT_Base_Uncased/vocab.txt", type=str)

parser.add_argument('--save_steps', default=100, type=int)
parser.add_argument('--check_steps', default=250, type=int)
parser.add_argument('--results_dir', default="results", type=str)

parser.add_argument('--is_position', default=False, type=bool)

cfg, unknown = parser.parse_known_args()


MAX_LENGTHS = {
    "SST": 128,
    "dbpedia": 256,
    "imdb": 128
}

NUM_LABELS = {
    "SST": 2,
    "dbpedia": 10,
    "imdb": 2
}

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(float(current) / float(rampup_length), 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, current_step, lambda_u, total_steps):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(current_step, total_steps)

class WeightEMA(object):
    def __init__(self, cfg, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.cfg = cfg

        params = list(model.state_dict().values())
        ema_params = list(ema_model.state_dict().values())

        self.params = list(map(lambda x: x.float(), params))
        self.ema_params = list(map(lambda x: x.float(), ema_params))
        self.wd = 0.02 * self.cfg.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())

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


def main():
    # Load Configuration
    model_cfg = configuration.model.from_json(cfg.model_cfg)        # BERT_cfg
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    #data = load_data(cfg)

    #if cfg.uda_mode or cfg.mixmatch_mode:
    #    data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
    #        else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    #else:
    #    data_iter = [data.sup_data_iter()]

    # my own implementation
    dataset = DataSet(cfg)
    train_dataset, val_dataset, unsup_dataset = dataset.get_dataset()

    # Create the DataLoaders for our training and validation sets.
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = cfg.train_batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = cfg.eval_batch_size # Evaluate with this batch size.
            )

    unsup_dataloader = None
    if unsup_dataset:
        unsup_dataloader = DataLoader(
            unsup_dataset,
            sampler = RandomSampler(unsup_dataset),
            batch_size = cfg.train_batch_size
        )

    if cfg.uda_mode or cfg.mixmatch_mode:
        data_iter = [train_dataloader, unsup_dataloader, validation_dataloader] 
    else:
        data_iter = [train_dataloader]

    ema_optimizer = None
    ema_model = None
    model = models.Classifier(model_cfg, NUM_LABELS[cfg.task])


    if cfg.uda_mode:
        if cfg.unsup_criterion == 'KL':
            unsup_criterion = nn.KLDivLoss(reduction='none')
        else:
            unsup_criterion = nn.MSELoss(reduction='none')
        sup_criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.optim4GPU(cfg, model)
    elif cfg.mixmatch_mode:
        train_criterion = SemiLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        ema_model = models.Classifier(model_cfg,  NUM_LABELS[cfg.task])
        for param in ema_model.parameters():
            param.detach_()
        ema_optimizer= WeightEMA(cfg, model, ema_model, alpha=cfg.ema_decay)
    else:
        sup_criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.optim4GPU(cfg, model)
    
    # Create trainer
    trainer = train.Trainer(cfg, model, data_iter, optimizer, get_device(), ema_model, ema_optimizer)

    # loss functions
    def get_label_guess_loss(model, sup_batch, unsup_batch, global_step):
        # batch
        input_ids, segment_ids, input_mask, label_ids, num_tokens = sup_batch
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask, \
        ori_num_tokens, aug_num_tokens = unsup_batch= unsup_batch


        all_ids = torch.cat([input_ids, ori_input_ids, aug_input_ids], dim=0)
        all_mask = torch.cat([input_mask, ori_input_mask, aug_input_mask], dim=0)
        all_seg = torch.cat([segment_ids, ori_segment_ids, aug_segment_ids], dim=0)

        all_logits = model(all_ids, all_seg, all_mask)
            
        #sup loss
        sup_size = label_ids.shape[0]
        sup_loss = sup_criterion(all_logits[:sup_size], label_ids)
        if cfg.tsa and cfg.tsa != "none":
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./all_logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        #unsup loss
        with torch.no_grad():
            outputs_u = model(ori_input_ids, ori_segment_ids, ori_input_mask)
            outputs_u2 = model(aug_input_ids, aug_segment_ids, aug_input_mask)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/cfg.uda_softmax_temp)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        targets_u = torch.cat([targets_u, targets_u], dim=0)

        # l2
        #probs_u = torch.softmax(all_logits[sup_size:], dim=1)
        #unsup_loss = torch.mean((probs_u - targets_u)**2)

        # kl
        aug_log_prob = F.log_softmax(all_logits[sup_size:], dim=-1)
        unsup_loss = torch.mean(torch.sum(unsup_criterion(aug_log_prob, targets_u), dim=-1))

        final_loss = sup_loss + cfg.uda_coeff*unsup_loss

        return final_loss, sup_loss, unsup_loss


    def get_loss_ict(model, sup_batch, unsup_batch, global_step):
        # batch
        input_ids, segment_ids, input_mask, og_label_ids, num_tokens = sup_batch
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask, \
        ori_num_tokens, aug_num_tokens = unsup_batch

        # convert label ids to hot vectors
        sup_size = input_ids.size(0)
        label_ids = torch.zeros(sup_size, 2).scatter_(1, og_label_ids.cpu().view(-1,1), 1)
        label_ids = label_ids.cuda(non_blocking=True)

        # sup mixup
        sup_l = np.random.beta(cfg.alpha, cfg.alpha)
        sup_l = max(sup_l, 1-sup_l)
        sup_idx = torch.randperm(sup_size)

        if cfg.sup_mixup == 'word' or cfg.sup_mixup == 'word_cls':
            if cfg.simple_pad:
                simple_pad(input_ids, input_mask, num_tokens)
            else:
                input_ids, c_input_ids = pad_for_word_mixup(
                    input_ids, input_mask, num_tokens, sup_idx
                )
                pdb.set_trace()
        else:
            c_input_ids = None

        # sup loss
        hidden = model(
            input_ids=input_ids, 
            segment_ids=segment_ids, 
            input_mask=input_mask,
            output_h=True,
            mixup=cfg.sup_mixup,
            shuffle_idx=sup_idx,
            clone_ids=c_input_ids,
            l=sup_l,
            manifold_mixup=cfg.manifold_mixup
        )
        logits = model(input_h=hidden)

        if cfg.sup_mixup:
            label_ids = mixup_op(label_ids, sup_l, sup_idx)

        sup_loss = -torch.sum(F.log_softmax(logits, dim=1) * label_ids, dim=1)

        if cfg.tsa and cfg.tsa != "none":
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(og_label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        if cfg.no_unsup_loss:
            return sup_loss, sup_loss, sup_loss, sup_loss

        # unsup loss
        with torch.no_grad():
            ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
            ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target


        # mixup
        l = np.random.beta(cfg.alpha, cfg.alpha)
        l = max(l, 1-l)
        idx = torch.randperm(hidden.size(0))

        
        if cfg.mixup == 'word' or cfg.mixup == 'word_cls':
            ori_input_ids, c_ori_input_ids = pad_for_word_mixup(
                ori_input_ids, ori_input_mask, ori_num_tokens, idx
            )
        else:
            c_ori_input_ids = None

        
        #for i in range(0, batch_size):
        #    new_mask = ori_input_mask[i]
        #    new_ids = ori_input_ids[i]
        #    old_ids = c_ori_input_ids[i]
        #    pdb.set_trace()

        hidden = model(
            input_ids=ori_input_ids, 
            segment_ids=ori_segment_ids, 
            input_mask=ori_input_mask,
            output_h=True,
            mixup=cfg.mixup,
            shuffle_idx=idx,
            clone_ids=c_ori_input_ids,
            l=l
        )
        logits = model(input_h=hidden)

        if cfg.mixup:
            ori_prob = mixup_op(ori_prob, l, idx)

        probs_u = torch.softmax(logits, dim=1)
        unsup_loss = torch.mean((probs_u - ori_prob)**2)

        w = cfg.uda_coeff * sigmoid_rampup(global_step, cfg.consistency_rampup_ends - cfg.consistency_rampup_starts)
        final_loss = sup_loss + w*unsup_loss
        return final_loss, sup_loss, unsup_loss, w*unsup_loss

    # evaluation
    def get_acc(model, batch):
        # input_ids, segment_ids, input_mask, label_id, sentence = batch
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)

        result = (label_pred == label_id).float()
        accuracy = result.mean()
        # output_dump.logs(sentence, label_pred, label_id)    # output dump

        return accuracy, result

    if cfg.mode == 'train':
        trainer.train(get_loss, None, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'train_eval':
        if cfg.mixmatch_mode:
            trainer.train(get_mixmatch_loss_short, get_acc, cfg.model_file, cfg.pretrain_file)
        elif cfg.uda_test_mode:
            trainer.train(get_sup_loss, get_acc, cfg.model_file, cfg.pretrain_file)
        elif cfg.uda_test_mode_two:
            trainer.train(get_loss_ict, get_acc, cfg.model_file, cfg.pretrain_file)
        else:
            trainer.train(get_loss_test, get_acc, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'eval':
        results = trainer.eval(get_acc, cfg.model_file, None)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy :' , total_accuracy)


if __name__ == '__main__':
    main()