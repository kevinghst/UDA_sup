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
import fire

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import train
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one
from utils import optim, configuration
import numpy as np

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
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


def main(cfg, model_cfg):
    # Load Configuration
    cfg = configuration.params.from_json(cfg)                   # Train or Eval cfg
    model_cfg = configuration.model.from_json(model_cfg)        # BERT_cfg
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    data = load_data(cfg)

    if cfg.uda_mode or cfg.mixmatch_mode:
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        data_iter = [data.sup_data_iter()]

    ema_optimizer = None
    ema_model = None
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))


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
        ema_model = models.Classifier(model_cfg, len(data.TaskDataset.labels))
        for param in ema_model.parameters():
            param.detach_()
        ema_optimizer= WeightEMA(cfg, model, ema_model, alpha=cfg.ema_decay)
    else:
        sup_criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.optim4GPU(cfg, model)
    
    # Create trainer
    trainer = train.Trainer(cfg, model, data_iter, optimizer, get_device(), ema_model, ema_optimizer)



    # Training
    def get_mixmatch_loss_two(model, sup_batch, unsup_batch, global_step):
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

        batch_size = input_ids.shape[0]
        sup_size = label_ids.shape[0]

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(input_ids=ori_input_ids, segment_ids=ori_segment_ids, input_mask=ori_input_mask)
            outputs_u2 = model(input_ids=aug_input_ids, segment_ids=aug_segment_ids, input_mask=aug_input_mask)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/cfg.uda_softmax_temp)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        input_ids = torch.cat((input_ids, ori_input_ids, aug_input_ids), dim=0)
        seg_ids = torch.cat((segment_ids, ori_segment_ids, aug_segment_ids), dim=0)
        input_mask = torch.cat((input_mask, ori_input_mask, aug_input_mask), dim=0)
        targets_u = torch.cat((targets_u, targets_u), dim=0)

        logits = model(input_ids, seg_ids, input_mask)

        logits_x = logits[:sup_size]
        logits_u = logits[sup_size:]

        sup_loss = sup_criterion(logits_x, label_ids)
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        probs_u = torch.softmax(logits_u, dim=1)
        unsup_loss = torch.sum(unsup_criterion(probs_u, targets_u), dim=-1)

        #unsup_loss = torch.mean((probs_u - targets_u)**2)

        final_loss = sup_loss + cfg.uda_coeff*unsup_loss
        return final_loss, sup_loss, unsup_loss
        


    def get_mixmatch_loss(model, sup_batch, unsup_batch, global_step):
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch

        batch_size = input_ids.shape[0]

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
        Lx, Lu, w = train_criterion(logits_x, targets_x, logits_u, targets_u, global_step, cfg.lambda_u, cfg.total_steps)

        loss = Lx + w * Lu
        return loss, Lx, Lu

    def get_loss(model, sup_batch, unsup_batch, global_step):
        # logits -> prob(softmax) -> log_prob(log_softmax)

        # batch
        input_ids, segment_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask = unsup_batch

            input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
            input_mask = torch.cat((input_mask, aug_input_mask), dim=0)
            
        # logits
        logits = model(input_ids, segment_ids, input_mask)

        # sup loss
        sup_size = label_ids.shape[0]            
        sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        if cfg.tsa:
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            # ori
            with torch.no_grad():
                ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
                ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target
                # ori_log_prob = F.log_softmax(ori_logits, dim=-1)

                # confidence-based masking
                if cfg.uda_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())
                    
            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
            aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

            # KLdiv loss
            """
                nn.KLDivLoss (kl_div)
                input : log_prob (log_softmax)
                target : prob    (softmax)
                https://pytorch.org/docs/stable/nn.html

                unsup_loss is divied by number of unsup_loss_mask
                it is different from the google UDA official
                The official unsup_loss is divided by total
                https://github.com/google-research/uda/blob/master/text/uda.py#L175
            """
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())
            final_loss = sup_loss + cfg.uda_coeff*unsup_loss

            return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

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
            trainer.train(get_mixmatch_loss, get_acc, cfg.model_file, cfg.pretrain_file)
        elif cfg.uda_test_mode:
            trainer.train(get_mixmatch_loss_two, get_acc, cfg.model_file, cfg.pretrain_file)
        elif cfg.uda_mode:
            trainer.train(get_loss, get_acc, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'eval':
        results = trainer.eval(get_acc, cfg.model_file, None)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy :' , total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
    #main('config/uda.json', 'config/bert_base.json')
